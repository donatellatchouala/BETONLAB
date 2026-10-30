"""BetonLab : laboratoire de résistance à la compression du béton.

Stockage :
  - PostgreSQL (permanent) si DATABASE_URL est défini dans les secrets ;
  - SQLite local (fichier betonlab_local.db) sinon, pour tester l'application.
Mode test : les essais de test vont dans une table séparée (beton_tests_demo),
les vrais essais (beton_tests) ne sont jamais touchés.
"""

import io
import math
import os
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")  # rendu sans écran (serveur)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import make_url


# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================

st.set_page_config(
    page_title="BetonLab",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#123B5D",
    "secondary": "#1D6A8A",
    "accent": "#16A085",
    "warning": "#F4B942",
    "danger": "#D9485F",
    "grid": "#DCE5EE",
}

# Les deux tables possibles : le nom vient toujours de cette liste,
# jamais d'une saisie de l'utilisateur (pas d'injection SQL possible).
TABLES = {
    "reel": "beton_tests",
    "test": "beton_tests_demo",
}

SPECIMEN_TYPES = ["Cube", "Cylindre"]
SQLITE_URL = "sqlite:///betonlab_local.db"
COLUMNS = [
    "id", "sample_code", "project", "test_date", "age_days", "specimen_type",
    "diameter_mm", "length_mm", "width_mm", "load_kn", "area_mm2",
    "strength_mpa", "notes", "created_at",
]


# ============================================================
# 2. INTERFACE GRAPHIQUE
# ============================================================
# Pour un rendu identique partout, ajoutez aussi le fichier
# .streamlit/config.toml fourni (thème clair imposé).

st.markdown(
    """
<style>
.stApp { background-color: #F3F6FB; color: #203040; }
h1, h2, h3, h4 { color: #123B5D !important; }
[data-testid="stSidebar"] { background-color: #FFFFFF; }
.stButton > button, .stFormSubmitButton > button {
    background-color: #123B5D; color: #FFFFFF !important;
    border-radius: 8px; border: none; font-weight: 600;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background-color: #16A085; color: #FFFFFF !important;
}
.stDownloadButton > button {
    background-color: #16A085; color: #FFFFFF !important; border-radius: 8px;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF; padding: 16px;
    border: 1px solid #DCE5EE; border-radius: 12px;
}
[data-testid="stMetricValue"] { color: #123B5D !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. OUTILS : SECRETS, MESSAGES, ERREURS
# ============================================================

def get_setting(name, default=None):
    """Lit un secret Streamlit, puis une variable d'environnement.

    st.secrets lève une exception s'il n'existe aucun fichier de secrets
    (cas d'un test en local) : on la capture au lieu de planter.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def postgres_driver():
    """Pilote PostgreSQL installé : psycopg2 en priorité, sinon psycopg 3.

    SQLAlchemy 2.1 utilise psycopg 3 par défaut pour « postgresql:// » :
    sans ce choix explicite, l'application plante si seul psycopg2 est
    installé (ce qui arrive après une mise à jour de SQLAlchemy).
    """
    for module, driver in (("psycopg2", "psycopg2"), ("psycopg", "psycopg")):
        try:
            __import__(module)
            return driver
        except ImportError:
            continue
    return None


def normalize_database_url(url):
    """Nettoie l'adresse et fixe le pilote PostgreSQL."""
    url = str(url).strip().strip('"').strip("'")
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            driver = postgres_driver()
            if driver is None:
                raise RuntimeError(
                    "Aucun pilote PostgreSQL installé : ajoutez "
                    "psycopg2-binary dans requirements.txt."
                )
            url = f"postgresql+{driver}://" + url[len(prefix):]
            break
    return url


def safe_error(error, url=None):
    """Message d'erreur court, sans jamais afficher le mot de passe."""
    message = str(error).strip().splitlines()[0] if str(error).strip() else ""
    if url:
        try:
            password = make_url(url).password
            if password:
                message = message.replace(password, "****")
        except Exception:
            pass
    return f"{type(error).__name__} : {message[:300]}"


def flash(kind, message):
    """Message affiché après un st.rerun() (sinon il disparaîtrait)."""
    st.session_state["_flash"] = (kind, message)


def show_flash():
    item = st.session_state.pop("_flash", None)
    if item:
        kind, message = item
        getattr(st, kind, st.info)(message)


# ============================================================
# 4. CONNEXION AU STOCKAGE (POSTGRESQL OU SQLITE)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine(database_url):
    if database_url.startswith("sqlite"):
        # check_same_thread=False : Streamlit utilise plusieurs threads.
        return create_engine(
            database_url, connect_args={"check_same_thread": False}
        )
    return create_engine(
        database_url,
        pool_pre_ping=True,      # reconnexion si Neon a mis la base en veille
        pool_recycle=1800,
        pool_size=3,
        max_overflow=2,
        connect_args={"connect_timeout": 10},
    )


def choose_storage():
    """Renvoie (url, nom lisible, permanent ?, remarque) selon la configuration.

    STORAGE_MODE (secret ou variable d'environnement) :
      "auto" (défaut) : PostgreSQL si DATABASE_URL existe, sinon SQLite ;
      "postgres"      : PostgreSQL obligatoire ;
      "sqlite"        : SQLite local, même si DATABASE_URL existe.
    """
    mode = str(get_setting("STORAGE_MODE", "auto")).lower().strip()
    database_url = get_setting("DATABASE_URL")
    use_local = st.session_state.get("_use_local_storage", False)

    if mode == "sqlite" or use_local or (mode == "auto" and not database_url):
        return SQLITE_URL, "SQLite local", False, ""

    if not database_url:
        st.error(
            "STORAGE_MODE vaut « postgres » mais DATABASE_URL est absent "
            "des secrets Streamlit."
        )
        st.stop()

    try:
        return normalize_database_url(database_url), "PostgreSQL", True, ""
    except RuntimeError:
        if mode == "postgres":
            st.error(
                "Aucun pilote PostgreSQL installé : ajoutez psycopg2-binary "
                "dans requirements.txt, puis redémarrez l'application."
            )
            st.stop()
        # Mode « auto » : l'application s'ouvre quand même, en stockage local.
        return SQLITE_URL, "SQLite local", False, (
            "PostgreSQL n'est pas disponible (pilote absent du serveur) : "
            "l'application utilise le stockage local. Pour garder les essais "
            "de façon permanente, ajoutez psycopg2-binary dans "
            "requirements.txt puis redémarrez l'application."
        )


def connect_storage():
    url, label, permanent, note = choose_storage()
    try:
        engine = get_engine(url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return engine, url, label, permanent, note
    except Exception as error:
        get_engine.clear()  # ne pas garder en cache un moteur défaillant
        st.error(
            "Connexion à la base impossible. Vérifie l'adresse, le mot de "
            "passe et que la base Neon est active.\n\n"
            f"Détail : {safe_error(error, url)}"
        )
        if label == "PostgreSQL":
            st.info(
                "Vous pouvez continuer en stockage local pour tester "
                "l'application. Attention : sur Streamlit Cloud, ce stockage "
                "est effacé à chaque redémarrage."
            )
            if st.button("Continuer en stockage local (test)", key="btn_local"):
                st.session_state["_use_local_storage"] = True
                st.rerun()
        st.stop()


# ============================================================
# 5. CRÉATION DES TABLES
# ============================================================

DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS {table} (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    sample_code VARCHAR(100) NOT NULL,
    project VARCHAR(200) DEFAULT 'Projet général',
    test_date DATE NOT NULL,
    age_days INTEGER NOT NULL,
    specimen_type VARCHAR(30) NOT NULL,
    diameter_mm DOUBLE PRECISION,
    length_mm DOUBLE PRECISION,
    width_mm DOUBLE PRECISION,
    load_kn DOUBLE PRECISION NOT NULL,
    area_mm2 DOUBLE PRECISION NOT NULL,
    strength_mpa DOUBLE PRECISION NOT NULL,
    notes TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
)
"""

DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS {table} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_code TEXT NOT NULL,
    project TEXT DEFAULT 'Projet général',
    test_date DATE NOT NULL,
    age_days INTEGER NOT NULL,
    specimen_type TEXT NOT NULL,
    diameter_mm REAL,
    length_mm REAL,
    width_mm REAL,
    load_kn REAL NOT NULL,
    area_mm2 REAL NOT NULL,
    strength_mpa REAL NOT NULL,
    notes TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


@st.cache_resource(show_spinner=False)
def init_database(database_url):
    """Exécuté une seule fois par serveur et par base (mis en cache)."""
    engine = get_engine(database_url)
    ddl = DDL_POSTGRES if engine.dialect.name == "postgresql" else DDL_SQLITE
    with engine.begin() as connection:
        for table in TABLES.values():
            connection.execute(text(ddl.format(table=table)))
            connection.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_date "
                f"ON {table} (test_date)"
            ))
    return True


# ============================================================
# 6. CALCULS
# ============================================================

def calculate_area(specimen_type, diameter=None, length=None, width=None):
    """Surface chargée en mm²."""
    if specimen_type == "Cylindre":
        if diameter is None or not diameter > 0:
            raise ValueError("Le diamètre doit être positif.")
        return math.pi * diameter ** 2 / 4
    if specimen_type == "Cube":
        if length is None or width is None or not (length > 0 and width > 0):
            raise ValueError("La longueur et la largeur doivent être positives.")
        return length * width
    raise ValueError(f"Type d'éprouvette inconnu : {specimen_type}")


def calculate_strength(load_kn, area_mm2):
    """Résistance en MPa : kN × 1000 / mm² = N/mm² = MPa."""
    if load_kn is None or not load_kn > 0:
        raise ValueError("La charge doit être positive.")
    if area_mm2 is None or not area_mm2 > 0:
        raise ValueError("La surface doit être positive.")
    return load_kn * 1000 / area_mm2


def parse_date(value):
    """Accepte un objet date, 2026-11-02 ou 02/11/2026."""
    if isinstance(value, date):
        return value
    text_value = str(value or "").strip()
    if not text_value or text_value.lower() in ("nan", "nat", "none"):
        raise ValueError("La date de l'essai est manquante.")
    try:
        return date.fromisoformat(text_value[:10])
    except ValueError:
        parsed = pd.to_datetime(text_value, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Date illisible : {text_value}") from None
        return parsed.date()


def build_record(sample_code, project, test_date, age_days, specimen_type,
                 diameter, length, width, load_kn, notes):
    """Vérifie une saisie et renvoie la ligne prête à enregistrer."""
    sample_code = str(sample_code or "").strip()
    if not sample_code:
        raise ValueError("Le code de l'éprouvette est obligatoire.")
    if len(sample_code) > 100:
        raise ValueError("Le code de l'éprouvette dépasse 100 caractères.")

    project = str(project or "").strip() or "Projet général"
    if len(project) > 200:
        raise ValueError("Le nom du projet dépasse 200 caractères.")

    test_date = parse_date(test_date)
    if test_date > date.today():
        raise ValueError("La date de l'essai est dans le futur.")

    if age_days is None:
        raise ValueError("L'âge du béton est manquant.")
    age_days = int(float(age_days))
    if not 0 <= age_days <= 3650:
        raise ValueError("L'âge doit être compris entre 0 et 3650 jours.")

    if specimen_type == "Cylindre":
        length = width = None
    else:
        diameter = None

    if load_kn is None:
        raise ValueError("La charge est manquante.")
    area = calculate_area(specimen_type, diameter, length, width)
    strength = calculate_strength(float(load_kn), area)

    if strength > 200:
        raise ValueError(
            f"Résistance calculée de {strength:.1f} MPa : invraisemblable. "
            "Vérifie la charge (en kN) et les dimensions (en mm)."
        )

    return {
        "sample_code": sample_code,
        "project": project,
        "test_date": test_date.isoformat(),
        "age_days": age_days,
        "specimen_type": specimen_type,
        "diameter_mm": float(diameter) if diameter is not None else None,
        "length_mm": float(length) if length is not None else None,
        "width_mm": float(width) if width is not None else None,
        "load_kn": float(load_kn),
        "area_mm2": float(area),
        "strength_mpa": float(strength),
        "notes": str(notes or "").strip(),
    }


# ============================================================
# 7. LECTURE ET ÉCRITURE DES ESSAIS
# ============================================================

INSERT_SQL = """
INSERT INTO {table} (
    sample_code, project, test_date, age_days, specimen_type,
    diameter_mm, length_mm, width_mm, load_kn, area_mm2, strength_mpa, notes
) VALUES (
    :sample_code, :project, :test_date, :age_days, :specimen_type,
    :diameter_mm, :length_mm, :width_mm, :load_kn, :area_mm2,
    :strength_mpa, :notes
)
"""


def save_tests(engine, table, records):
    """Enregistre une ou plusieurs lignes dans une seule transaction."""
    if not records:
        return 0
    with engine.begin() as connection:
        connection.execute(text(INSERT_SQL.format(table=table)), records)
    fetch_tests.clear()
    return len(records)


def existing_codes(engine, table):
    with engine.connect() as connection:
        rows = connection.execute(text(f"SELECT sample_code FROM {table}"))
        return {str(r[0]).strip().lower() for r in rows}


def delete_tests(engine, table, ids):
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    statement = text(f"DELETE FROM {table} WHERE id IN :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    with engine.begin() as connection:
        result = connection.execute(statement, {"ids": ids})
    fetch_tests.clear()
    return result.rowcount


def delete_all(engine, table):
    with engine.begin() as connection:
        connection.execute(text(f"DELETE FROM {table}"))
    fetch_tests.clear()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tests(database_url, table):
    """Lit tous les essais. Mis en cache 5 min, vidé après chaque écriture."""
    engine = get_engine(database_url)
    with engine.connect() as connection:
        df = pd.read_sql(
            text(
                f"SELECT {', '.join(COLUMNS)} FROM {table} "
                "ORDER BY test_date DESC, id DESC"
            ),
            connection,
        )
    # SQLite renvoie les dates en texte, PostgreSQL en objets date :
    # on uniformise pour que filtres et graphiques marchent dans les deux cas.
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce").dt.date
    for column in ["age_days", "diameter_mm", "length_mm", "width_mm",
                   "load_kn", "area_mm2", "strength_mpa"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["notes"] = df["notes"].fillna("")
    return df


# ============================================================
# 8. DONNÉES DE TEST ET IMPORT CSV
# ============================================================

def demo_records(n=20, start=1, seed=None):
    """Essais fictifs mais réalistes (béton C25/30) pour tester l'application."""
    rng = np.random.default_rng(seed)
    records = []
    today = date.today()
    for i in range(start, start + n):
        specimen = "Cube" if i % 3 else "Cylindre"
        age = int(rng.choice([7, 14, 28, 28, 28]))
        # Montée en résistance approximative avec l'âge.
        factor = {7: 0.67, 14: 0.88, 28: 1.0}[age]
        target = 38.0 if specimen == "Cube" else 31.0
        strength = max(5.0, rng.normal(target * factor, 3.0))
        if specimen == "Cylindre":
            diameter, length, width = 160.0, None, None
            area = math.pi * diameter ** 2 / 4
        else:
            diameter, length, width = None, 150.0, 150.0
            area = length * width
        load = round(strength * area / 1000, 1)
        records.append(build_record(
            f"DEMO-{i:03d}", "Projet de test",
            today - timedelta(days=int(rng.integers(0, 60))), age, specimen,
            diameter, length, width, load, "Donnée générée pour les tests",
        ))
    return records


def parse_import(file_bytes, known_codes):
    """Lit un CSV (format de l'export BetonLab) et renvoie (lignes, erreurs).

    La surface et la résistance sont recalculées : on ne fait pas confiance
    aux valeurs du fichier.
    """
    raw = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=None,
                              engine="python", encoding=encoding)
            break
        except Exception:
            continue
    if raw is None:
        return [], ["Fichier illisible : enregistrez-le en CSV (UTF-8)."]

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = {"sample_code", "test_date", "age_days",
                "specimen_type", "load_kn"}
    missing = required - set(raw.columns)
    if missing:
        return [], [f"Colonnes manquantes : {', '.join(sorted(missing))}."]

    def value(row, column):
        v = row.get(column)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

    def number(row, column):
        v = value(row, column)
        if v is None or str(v).strip() == "":
            return None
        return float(str(v).replace(",", "."))

    records, errors, seen = [], [], set(known_codes)
    for index, row in raw.iterrows():
        line = index + 2  # +1 pour l'en-tête, +1 car Excel commence à 1
        try:
            code = str(value(row, "sample_code") or "").strip()
            if code.lower() in seen:
                errors.append(f"Ligne {line} : code {code} déjà présent, ignorée.")
                continue
            specimen = str(value(row, "specimen_type") or "").strip().capitalize()
            record = build_record(
                code, value(row, "project"),
                str(value(row, "test_date"))[:10], number(row, "age_days"),
                specimen, number(row, "diameter_mm"), number(row, "length_mm"),
                number(row, "width_mm"), number(row, "load_kn"),
                value(row, "notes"),
            )
            records.append(record)
            seen.add(code.lower())
        except Exception as error:
            errors.append(f"Ligne {line} : {error}")
    return records, errors


# ============================================================
# 9. FILTRES COMMUNS AUX ANALYSES
# ============================================================

def filter_ui(df, key):
    """Filtres projet / type / âge. Mélanger cubes et cylindres fausse
    les statistiques (un cube donne environ 20 % de plus)."""
    with st.expander("Filtres", expanded=True):
        c1, c2, c3 = st.columns(3)
        projects = sorted(df["project"].dropna().astype(str).unique())
        chosen_projects = c1.multiselect(
            "Projets", projects, default=projects, key=f"{key}_projects")
        types = sorted(df["specimen_type"].dropna().astype(str).unique())
        default_type = types[0] if len(types) == 1 else "Tous"
        chosen_type = c2.selectbox(
            "Type d'éprouvette", ["Tous"] + types,
            index=(["Tous"] + types).index(default_type), key=f"{key}_type")
        ages = sorted(int(a) for a in df["age_days"].dropna().unique())
        chosen_ages = c3.multiselect(
            "Âges (jours)", ages, default=ages, key=f"{key}_ages")

    view = df[df["project"].astype(str).isin(chosen_projects)
              & df["age_days"].isin(chosen_ages)]
    if chosen_type != "Tous":
        view = view[view["specimen_type"] == chosen_type]
    elif view["specimen_type"].nunique() > 1:
        st.warning(
            "Cubes et cylindres sont mélangés : choisissez un seul type "
            "pour des statistiques comparables."
        )
    return view


# ============================================================
# 10. DÉMARRAGE
# ============================================================

engine, DB_URL, STORAGE_LABEL, PERMANENT, STORAGE_NOTE = connect_storage()

try:
    init_database(DB_URL)
except Exception as err:
    init_database.clear()
    st.error(
        "Impossible de créer les tables. Vérifie les droits PostgreSQL.\n\n"
        f"Détail : {safe_error(err, DB_URL)}"
    )
    st.stop()


# ============================================================
# 11. MENU DE NAVIGATION ET MODE TEST
# ============================================================

st.sidebar.title("🏗️ BetonLab")
st.sidebar.caption("Laboratoire de résistance du béton")
st.sidebar.divider()

test_mode = st.sidebar.toggle(
    "🧪 Mode test",
    key="test_mode",
    help="Les essais saisis en mode test vont dans une table séparée. "
         "Les vrais essais ne sont ni lus ni modifiés.",
)
TABLE = TABLES["test" if test_mode else "reel"]

page = st.sidebar.radio(
    "Navigation",
    [
        "Tableau de bord", "Enregistrer un essai", "Base des essais",
        "Statistiques", "Probabilités", "Diagrammes", "Historique",
        "Paramètres",
    ],
    key="page",
)

st.sidebar.divider()
st.sidebar.caption(
    f"Stockage : **{STORAGE_LABEL}**"
    + ("" if PERMANENT else " (temporaire sur Streamlit Cloud)")
)
st.sidebar.caption(f"Table : `{TABLE}`")

if st.sidebar.button("🔄 Réinitialiser l'interface", key="btn_reset_side",
                     width="stretch"):
    keep = {"_use_local_storage"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.rerun()

st.sidebar.caption("La réinitialisation de l'interface ne supprime pas les essais.")

if test_mode:
    st.info(
        "🧪 **Mode test actif** : vous travaillez sur la table "
        f"`{TABLE}`. Les vrais essais ne sont pas affectés."
    )
if STORAGE_NOTE:
    st.warning(STORAGE_NOTE)
elif not PERMANENT:
    st.warning(
        "Stockage local SQLite : parfait pour tester, mais sur Streamlit "
        "Cloud ce fichier est effacé à chaque redémarrage. Pour garder les "
        "essais, configurez DATABASE_URL (PostgreSQL)."
    )

show_flash()

try:
    df = fetch_tests(DB_URL, TABLE)
except Exception as err:
    fetch_tests.clear()
    st.error(f"Impossible de lire les essais. Détail : {safe_error(err, DB_URL)}")
    st.stop()


# ============================================================
# 12. TABLEAU DE BORD
# ============================================================

if page == "Tableau de bord":

    st.title("🏗️ BetonLab")
    st.subheader("Analyse de la résistance à la compression du béton")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre d'essais", len(df))

    values = df["strength_mpa"].dropna()
    if values.empty:
        st.info(
            "Aucun essai enregistré pour le moment."
            + (" En mode test, générez des essais dans « Paramètres »."
               if test_mode else "")
        )
    else:
        c2.metric("Résistance moyenne", f"{values.mean():.2f} MPa")
        c3.metric("Résistance minimale", f"{values.min():.2f} MPa")
        c4.metric("Résistance maximale", f"{values.max():.2f} MPa")
        st.subheader("Derniers essais enregistrés")
        st.dataframe(df.head(10), width="stretch", hide_index=True)


# ============================================================
# 13. ENREGISTRER UN ESSAI
# ============================================================

elif page == "Enregistrer un essai":

    st.title("🧪 Enregistrer un essai")

    # Hors du formulaire : dans un st.form, changer ce choix ne rafraîchit
    # pas la page, et les bons champs (diamètre ou côtés) n'apparaissaient pas.
    specimen_type = st.radio(
        "Type d'éprouvette", SPECIMEN_TYPES, horizontal=True, key="specimen")

    with st.form("beton_test_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            sample_code = st.text_input("Code de l'éprouvette",
                                        placeholder="BET-001", max_chars=100)
            project = st.text_input("Projet", value="Projet général",
                                    max_chars=200)
            test_date = st.date_input("Date de l'essai", value=date.today(),
                                      max_value=date.today(), format="DD/MM/YYYY")
            age_days = st.number_input("Âge du béton (jours)", min_value=0,
                                       max_value=3650, value=28, step=1)

        with col2:
            diameter = length = width = None
            if specimen_type == "Cylindre":
                diameter = st.number_input("Diamètre (mm)", min_value=1.0,
                                           value=160.0, step=1.0)
            else:
                length = st.number_input("Longueur de la face chargée (mm)",
                                         min_value=1.0, value=150.0, step=1.0)
                width = st.number_input("Largeur de la face chargée (mm)",
                                        min_value=1.0, value=150.0, step=1.0)
            load_kn = st.number_input("Charge maximale (kN)", min_value=0.0,
                                      value=750.0, step=1.0)
            notes = st.text_area("Observations")

        submitted = st.form_submit_button("Calculer et enregistrer")

    if submitted:
        try:
            record = build_record(sample_code, project, test_date, age_days,
                                  specimen_type, diameter, length, width,
                                  load_kn, notes)
            if record["sample_code"].lower() in existing_codes(engine, TABLE):
                raise ValueError(
                    f"Le code {record['sample_code']} existe déjà. "
                    "Choisissez un code unique (ex. BET-002)."
                )
            save_tests(engine, TABLE, [record])
            st.success("Essai enregistré avec succès.")
            m1, m2 = st.columns(2)
            m1.metric("Résistance calculée", f"{record['strength_mpa']:.2f} MPa")
            m2.metric("Surface chargée", f"{record['area_mm2']:.0f} mm²")
        except ValueError as error:
            st.error(f"Erreur : {error}")
        except Exception as error:
            st.error(f"Enregistrement impossible. Détail : {safe_error(error, DB_URL)}")


# ============================================================
# 14. BASE DES ESSAIS, EXPORT ET SUPPRESSION
# ============================================================

elif page == "Base des essais":

    st.title("🗄️ Base des essais")

    search = st.text_input("Rechercher par code ou projet", key="search")
    view = df
    if search.strip():
        mask = (
            view["sample_code"].astype(str).str.contains(search, case=False, na=False, regex=False)
            | view["project"].astype(str).str.contains(search, case=False, na=False, regex=False)
        )
        view = view[mask]

    st.caption(f"{len(view)} essai(s) affiché(s) sur {len(df)}.")
    st.dataframe(view, width="stretch", hide_index=True)

    st.download_button(
        "📥 Exporter en CSV (sauvegarde réimportable)",
        data=view.to_csv(index=False, sep=";").encode("utf-8-sig"),
        file_name=f"betonlab_{TABLE}_{date.today().isoformat()}.csv",
        mime="text/csv",
        key="btn_export",
    )

    if not df.empty:
        st.divider()
        st.subheader("Supprimer des essais")
        labels = {int(r.id): f"#{int(r.id)} · {r.sample_code} · {r.project}"
                  for r in df.itertuples()}
        to_delete = st.multiselect("Essais à supprimer", list(labels),
                                   format_func=labels.get, key="to_delete")
        if st.button("🗑️ Supprimer la sélection", key="btn_delete_sel",
                     disabled=not to_delete):
            try:
                n = delete_tests(engine, TABLE, to_delete)
                flash("success", f"{n} essai(s) supprimé(s).")
                st.session_state.pop("to_delete", None)
                st.rerun()
            except Exception as error:
                st.error(f"Suppression impossible. Détail : {safe_error(error, DB_URL)}")


# ============================================================
# 15. STATISTIQUES
# ============================================================

elif page == "Statistiques":

    st.title("📊 Statistiques descriptives")

    if df.empty:
        st.info("Enregistrez d'abord des essais.")
        st.stop()

    view = filter_ui(df, "stats")
    values = view["strength_mpa"].dropna()
    if values.empty:
        st.warning("Aucun essai ne correspond aux filtres.")
        st.stop()

    n = len(values)
    mean = values.mean()
    sd = values.std(ddof=1) if n > 1 else np.nan
    cv = sd / mean * 100 if n > 1 and mean != 0 else np.nan

    results = pd.DataFrame({
        "Indicateur": [
            "Nombre d'essais", "Moyenne (MPa)", "Médiane (MPa)",
            "Minimum (MPa)", "Maximum (MPa)",
            "Écart-type expérimental (MPa)", "Coefficient de variation (%)",
        ],
        "Valeur": [n, mean, values.median(), values.min(), values.max(), sd, cv],
    })
    results["Valeur"] = [str(n)] + [
        "—" if pd.isna(v) else f"{v:.2f}" for v in results["Valeur"][1:]
    ]
    st.dataframe(results, width="stretch", hide_index=True)

    if n > 1:
        st.subheader("Par âge")
        by_age = (view.groupby("age_days")["strength_mpa"]
                  .agg(["count", "mean", "std", "min", "max"]).round(2))
        by_age.index.name = "Âge (jours)"
        by_age.columns = ["Nombre", "Moyenne", "Écart-type", "Min", "Max"]
        st.dataframe(by_age, width="stretch")


# ============================================================
# 16. PROBABILITÉS
# ============================================================

elif page == "Probabilités":

    st.title("🎯 Analyse des probabilités")

    if df.empty:
        st.info("Enregistrez d'abord des essais.")
        st.stop()

    view = filter_ui(df, "proba")
    values = view["strength_mpa"].dropna().to_numpy(dtype=float)

    if len(values) < 2:
        st.warning("Au moins deux résultats sont nécessaires pour estimer la dispersion.")
        st.stop()

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))

    c1, c2, c3 = st.columns(3)
    c1.metric("Moyenne", f"{mean:.2f} MPa")
    c2.metric("Écart-type", f"{sd:.2f} MPa")

    if sd <= 0:
        st.warning("L'écart-type est nul : la loi normale n'est pas exploitable.")
        st.stop()

    fk = mean - 1.645 * sd
    c3.metric("Fractile 5 % estimé", f"{fk:.2f} MPa",
              help="m − 1,645·s : valeur dépassée par 95 % des éprouvettes "
                   "sous l'hypothèse d'une loi normale.")

    threshold = st.number_input("Résistance seuil (MPa)", min_value=0.0,
                                value=round(max(mean, 0.0), 1), step=0.5,
                                key="threshold")
    p_below = float(norm.cdf(threshold, loc=mean, scale=sd))
    d1, d2 = st.columns(2)
    d1.metric("Probabilité estimée de dépasser le seuil", f"{(1 - p_below) * 100:.2f} %")
    d2.metric("Probabilité estimée d'être sous le seuil", f"{p_below * 100:.2f} %")

    if len(values) < 15:
        st.caption(f"Seulement {len(values)} résultats : les probabilités sont très incertaines.")
    st.info(
        "Estimations sous hypothèse de distribution normale. Elles ne "
        "constituent pas une vérification de conformité selon la norme."
    )


# ============================================================
# 17. DIAGRAMMES
# ============================================================

elif page == "Diagrammes":

    st.title("📈 Diagrammes des essais")

    if df.empty:
        st.info("Aucune donnée disponible.")
        st.stop()

    view = filter_ui(df, "diag")
    values = view["strength_mpa"].dropna()
    if values.empty:
        st.warning("Aucun essai ne correspond aux filtres.")
        st.stop()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(values, bins=min(12, max(3, int(np.sqrt(len(values))))),
            color=COLORS["secondary"], edgecolor="white")
    ax.axvline(values.mean(), color=COLORS["danger"], linestyle="--",
               label=f"Moyenne {values.mean():.1f} MPa")
    ax.set_title("Distribution des résistances")
    ax.set_xlabel("Résistance (MPa)")
    ax.set_ylabel("Nombre d'éprouvettes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(view["age_days"], view["strength_mpa"],
               color=COLORS["accent"], alpha=0.8)
    ax.set_title("Résistance en fonction de l'âge")
    ax.set_xlabel("Âge (jours)")
    ax.set_ylabel("Résistance (MPa)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    last = view.sort_values(["test_date", "id"]).tail(30)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(last["sample_code"].astype(str), last["strength_mpa"],
           color=COLORS["primary"])
    ax.set_title("Résistance par éprouvette (30 derniers essais)")
    ax.set_xlabel("Éprouvette")
    ax.set_ylabel("Résistance (MPa)")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 18. HISTORIQUE
# ============================================================

elif page == "Historique":

    st.title("📚 Historique des essais")

    dates = df["test_date"].dropna()
    if dates.empty:
        st.info("Aucun historique disponible.")
        st.stop()

    c1, c2 = st.columns(2)
    start_date = c1.date_input("Date de début", value=min(dates),
                               format="DD/MM/YYYY", key="hist_start")
    end_date = c2.date_input("Date de fin", value=max(dates),
                             format="DD/MM/YYYY", key="hist_end")

    if start_date > end_date:
        st.error("La date de début doit précéder la date de fin.")
    else:
        dated = df[df["test_date"].notna()]
        history = dated[(dated["test_date"] >= start_date)
                        & (dated["test_date"] <= end_date)]
        st.dataframe(history, width="stretch", hide_index=True)
        st.write(f"Nombre d'essais : {len(history)}")


# ============================================================
# 19. PARAMÈTRES : STOCKAGE, DONNÉES DE TEST, IMPORT, SUPPRESSION
# ============================================================

elif page == "Paramètres":

    st.title("⚙️ Paramètres de BetonLab")

    st.subheader("Stockage des données")
    s1, s2, s3 = st.columns(3)
    s1.metric("Stockage", STORAGE_LABEL)
    s2.metric("Table utilisée", "test" if test_mode else "réelle")
    s3.metric("Essais dans cette table", len(df))
    st.caption(
        "Pour choisir le stockage, ajoutez STORAGE_MODE dans les secrets : "
        "« auto » (défaut), « postgres » ou « sqlite »."
    )

    st.divider()
    st.subheader("🧪 Données de test")
    if not test_mode:
        st.write(
            "Activez le **Mode test** dans le menu de gauche pour générer et "
            "supprimer des essais de test, sans toucher aux vrais essais."
        )
    else:
        n_demo = st.slider("Nombre d'essais à générer", 5, 100, 20, key="n_demo")
        g1, g2 = st.columns(2)
        if g1.button("Générer des essais de test", key="btn_demo"):
            try:
                known = existing_codes(engine, TABLE)
                numbers = [int(c[5:]) for c in known
                           if c.startswith("demo-") and c[5:].isdigit()]
                records = demo_records(n_demo, start=max(numbers, default=0) + 1)
                saved = save_tests(engine, TABLE, records)
                flash("success", f"{saved} essais de test ajoutés.")
                st.rerun()
            except Exception as error:
                st.error(f"Génération impossible. Détail : {safe_error(error, DB_URL)}")
        if g2.button("Vider la table de test", key="btn_clear_demo"):
            try:
                delete_all(engine, TABLES["test"])
                flash("success", "Table de test vidée.")
                st.rerun()
            except Exception as error:
                st.error(f"Suppression impossible. Détail : {safe_error(error, DB_URL)}")

    st.divider()
    st.subheader("📤 Importer des essais (CSV)")
    st.write(
        "Format : celui de l'export (« Base des essais »). Colonnes "
        "obligatoires : sample_code, test_date, age_days, specimen_type, "
        "load_kn, et diameter_mm (cylindre) ou length_mm et width_mm (cube). "
        "La surface et la résistance sont recalculées."
    )
    uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="upload_csv")
    if uploaded is not None:
        records, errors = parse_import(uploaded.getvalue(), existing_codes(engine, TABLE))
        st.write(f"{len(records)} ligne(s) valide(s), {len(errors)} problème(s).")
        if errors:
            with st.expander("Voir les problèmes"):
                for e in errors[:200]:
                    st.write("• " + e)
        if records and st.button(f"Importer {len(records)} essai(s) dans la table "
                                 f"{'de test' if test_mode else 'réelle'}",
                                 key="btn_import"):
            try:
                n = save_tests(engine, TABLE, records)
                flash("success", f"{n} essai(s) importé(s).")
                st.session_state.pop("upload_csv", None)
                st.rerun()
            except Exception as error:
                st.error(f"Import impossible. Détail : {safe_error(error, DB_URL)}")

    st.divider()
    st.subheader("Suppression de tous les essais de cette table")
    st.warning(
        "Opération irréversible : exportez d'abord une sauvegarde CSV. "
        f"Table concernée : `{TABLE}`."
    )
    confirm_delete = st.checkbox(
        "Je confirme vouloir supprimer TOUS les essais de cette table.",
        key="confirm_delete")
    confirm_phrase = st.text_input("Écrivez SUPPRIMER pour confirmer",
                                   key="confirm_phrase")
    if st.button("🗑️ Supprimer définitivement tous les essais",
                 type="primary", key="btn_delete_all"):
        if confirm_delete and confirm_phrase.strip() == "SUPPRIMER":
            try:
                delete_all(engine, TABLE)
                for k in ("confirm_delete", "confirm_phrase"):
                    st.session_state.pop(k, None)
                flash("success", "Tous les essais de cette table ont été supprimés.")
                st.rerun()
            except Exception as error:
                st.error(f"La suppression a échoué. Détail : {safe_error(error, DB_URL)}")
        else:
            st.error("Cochez la confirmation et écrivez SUPPRIMER.")


# ============================================================
# 20. PIED DE PAGE
# ============================================================

st.sidebar.divider()
st.sidebar.caption("BetonLab • Analyse expérimentale du béton")
st.sidebar.caption(
    "Vérifiez les unités et les normes applicables avant toute "
    "utilisation professionnelle."
)
