import os
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import URL


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BetonLab | Analyse du béton",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "betonlab.db")


def lire_url_base():
    """Lit l'URL PostgreSQL dans les secrets ou les variables système."""
    url = os.getenv("DATABASE_URL", "").strip()

    try:
        url = st.secrets.get("DATABASE_URL", url)
        if not url and "database" in st.secrets:
            url = st.secrets["database"].get("url", "")
    except Exception:
        pass

    url = str(url).strip()

    if url.startswith("postgres://"):
        url = url.replace(
            "postgres://", "postgresql+psycopg://", 1
        )
    elif url.startswith("postgresql://"):
        url = url.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )

    return url


DATABASE_URL = lire_url_base()


@st.cache_resource
def creer_moteur(url):
    """Crée une connexion réutilisable à la base de données."""
    if url.startswith("sqlite:"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True
        )

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800
    )


if DATABASE_URL:
    ENGINE = creer_moteur(DATABASE_URL)
    TYPE_BASE = ENGINE.dialect.name
else:
    ENGINE = creer_moteur("sqlite:///" + DB_PATH)
    TYPE_BASE = "sqlite"


# =========================================================
# 2. THEMES
# =========================================================

THEMES = {
    "Clair": {
        "fond": "#FFFFFF",
        "panneau": "#F2F5FA",
        "texte": "#172B4D",
        "secondaire": "#64748B",
        "primaire": "#173B67",
        "accent": "#087F8C",
        "bordure": "#D9E2EF",
        "danger": "#D64545",
        "graphique": "plotly_white"
    },
    "Sombre": {
        "fond": "#000000",
        "panneau": "#171717",
        "texte": "#FFFFFF",
        "secondaire": "#C1C7D0",
        "primaire": "#8CB8FF",
        "accent": "#38D6C5",
        "bordure": "#444444",
        "danger": "#FF7070",
        "graphique": "plotly_dark"
    }
}

if "theme_betonlab" not in st.session_state:
    st.session_state.theme_betonlab = "Clair"


# =========================================================
# 3. INITIALISATION ET MIGRATION DE LA BASE
# =========================================================

COLONNES_STANDARD = {
    "project": "TEXT NOT NULL DEFAULT 'Projet sans nom'",
    "sample_code": "TEXT DEFAULT ''",
    "test_date": "TEXT DEFAULT '2000-01-01'",
    "age_days": "INTEGER DEFAULT 28",
    "length_mm": "REAL DEFAULT 150",
    "width_mm": "REAL DEFAULT 150",
    "height_mm": "REAL DEFAULT 300",
    "load_kn": "REAL DEFAULT 1",
    "strength_mpa": "REAL DEFAULT 0",
    "shape": "TEXT DEFAULT 'Prismatique'"
}


def initialiser_base():
    """Crée les tables et ajoute les colonnes absentes sans supprimer les essais."""

    type_id = (
        "BIGSERIAL PRIMARY KEY"
        if TYPE_BASE == "postgresql"
        else "INTEGER PRIMARY KEY AUTOINCREMENT"
    )

    with ENGINE.begin() as con:
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS tests (
                id {type_id},
                project TEXT NOT NULL DEFAULT 'Projet sans nom',
                sample_code TEXT DEFAULT '',
                test_date TEXT DEFAULT '2000-01-01',
                age_days INTEGER DEFAULT 28,
                length_mm REAL DEFAULT 150,
                width_mm REAL DEFAULT 150,
                height_mm REAL DEFAULT 300,
                load_kn REAL DEFAULT 1,
                strength_mpa REAL DEFAULT 0,
                shape TEXT DEFAULT 'Prismatique'
            )
        """))

        colonnes = {
            c["name"]
            for c in inspect(ENGINE).get_columns("tests")
        }

        colonnes_ajoutees = set()

        for nom, definition in COLONNES_STANDARD.items():
            if nom not in colonnes:
                con.execute(text(
                    f"ALTER TABLE tests ADD COLUMN {nom} {definition}"
                ))
                colonnes_ajoutees.add(nom)

        # Reprendre quelques noms de colonnes des anciennes versions.
        colonnes = {
            c["name"]
            for c in inspect(ENGINE).get_columns("tests")
        }

        correspondances = {
            "project": ["projet"],
            "sample_code": ["code", "sample"],
            "test_date": ["date", "date_essai"],
            "age_days": ["age", "age_beton"],
            "length_mm": ["length", "longueur", "diametre"],
            "width_mm": ["width", "largeur"],
            "height_mm": ["height", "hauteur"],
            "load_kn": ["charge", "charge_kn"],
            "strength_mpa": ["resistance", "resistance_mpa"],
            "shape": ["forme"]
        }

        for nouvelle, anciennes in correspondances.items():
            if nouvelle not in colonnes_ajoutees:
                continue

            for ancienne in anciennes:
                if ancienne in colonnes:
                    con.execute(text(
                        f"""
                        UPDATE tests
                        SET {nouvelle} = {ancienne}
                        WHERE {ancienne} IS NOT NULL
                        """
                    ))
                    break

        con.execute(text("""
            CREATE TABLE IF NOT EXISTS historique (
                id INTEGER PRIMARY KEY,
                date_heure TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT DEFAULT ''
            )
        """))

        if TYPE_BASE == "postgresql":
            con.execute(text("""
                CREATE SEQUENCE IF NOT EXISTS historique_id_seq
            """))
            con.execute(text("""
                ALTER TABLE historique
                ALTER COLUMN id SET DEFAULT nextval('historique_id_seq')
            """))
            con.execute(text("""
                SELECT setval(
                    'historique_id_seq',
                    GREATEST(
                        COALESCE((SELECT MAX(id) FROM historique), 0) + 1,
                        1
                    ),
                    false
                )
            """))

        con.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tests_date
            ON tests(test_date)
        """))


def enregistrer_action(action, details=""):
    with ENGINE.begin() as con:
        con.execute(text("""
            INSERT INTO historique(date_heure, action, details)
            VALUES (:date_heure, :action, :details)
        """), {
            "date_heure": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "action": action,
            "details": details
        })


def charger_essais():
    with ENGINE.connect() as con:
        df = pd.read_sql_query(
            text("""
                SELECT id, project, sample_code, test_date, age_days,
                       length_mm, width_mm, height_mm,
                       load_kn, strength_mpa, shape
                FROM tests
                ORDER BY test_date DESC, id DESC
            """),
            con
        )

    if not df.empty:
        df["project"] = (
            df["project"].fillna("Projet sans nom").astype(str)
        )
        df["shape"] = (
            df["shape"].fillna("Prismatique").astype(str)
        )
        df["test_date"] = pd.to_datetime(
            df["test_date"], errors="coerce"
        )

    return df


def charger_historique():
    with ENGINE.connect() as con:
        return pd.read_sql_query(
            text("""
                SELECT
                    date_heure AS "Date et heure",
                    action AS "Action",
                    details AS "Détails"
                FROM historique
                ORDER BY id DESC
            """),
            con
        )


# =========================================================
# 4. MIGRATION OPTIONNELLE DE L'ANCIEN FICHIER SQLITE
# =========================================================

def migrer_sqlite_local_vers_distant():
    """
    Si une ancienne base SQLite existe encore et que PostgreSQL
    est vide, copie ses essais dans la base distante.
    Les essais distants existants ne sont jamais écrasés.
    """
    if TYPE_BASE == "sqlite" or not os.path.exists(DB_PATH):
        return

    try:
        with ENGINE.connect() as con:
            nombre_distant = con.execute(
                text("SELECT COUNT(*) FROM tests")
            ).scalar_one()

        if nombre_distant > 0:
            return

        local = sqlite3.connect(DB_PATH)
        local.row_factory = sqlite3.Row

        table = local.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'tests'
        """).fetchone()

        if table is None:
            local.close()
            return

        lignes = local.execute("SELECT * FROM tests").fetchall()
        local.close()

        if not lignes:
            return

        compte = 0

        with ENGINE.begin() as con:
            for ligne in lignes:
                r = dict(ligne)

                def lire(*noms, defaut=None):
                    for nom in noms:
                        if nom in r and r[nom] is not None:
                            return r[nom]
                    return defaut

                charge = float(lire("load_kn", "charge", defaut=1))
                forme = str(lire("shape", "forme", defaut="Prismatique"))
                longueur = float(lire("length_mm", "longueur", "length", defaut=150))
                largeur = float(lire("width_mm", "largeur", "width", defaut=150))

                resistance = lire(
                    "strength_mpa", "resistance", "resistance_mpa"
                )

                if resistance is None:
                    surface = (
                        np.pi * longueur**2 / 4
                        if forme == "Cylindrique"
                        else longueur * largeur
                    )
                    resistance = charge * 1000 / surface

                con.execute(text("""
                    INSERT INTO tests (
                        project, sample_code, test_date, age_days,
                        length_mm, width_mm, height_mm,
                        load_kn, strength_mpa, shape
                    )
                    VALUES (
                        :project, :sample_code, :test_date, :age_days,
                        :length_mm, :width_mm, :height_mm,
                        :load_kn, :strength_mpa, :shape
                    )
                """), {
                    "project": lire("project", "projet", defaut="Projet importé"),
                    "sample_code": lire("sample_code", "code", defaut="Sans code"),
                    "test_date": str(lire("test_date", "date", defaut="2000-01-01")),
                    "age_days": int(lire("age_days", "age", defaut=28)),
                    "length_mm": longueur,
                    "width_mm": largeur,
                    "height_mm": float(lire("height_mm", "hauteur", "height", defaut=300)),
                    "load_kn": charge,
                    "strength_mpa": float(resistance),
                    "shape": forme
                })

                compte += 1

        if compte:
            enregistrer_action(
                "Migration de données",
                f"{compte} essai(s) copié(s) depuis SQLite"
            )

    except Exception as erreur:
        st.warning(
            "La migration automatique de l'ancienne base SQLite "
            f"n'a pas abouti : {erreur}. Utilisez l'importation CSV."
        )


# Initialiser la base et récupérer les données
try:
    initialiser_base()
    migrer_sqlite_local_vers_distant()
except Exception as erreur:
    st.error(
        "Impossible d'initialiser la base de données. "
        f"Vérifiez DATABASE_URL et les droits d'accès. Détail : {erreur}"
    )
    st.stop()


# =========================================================
# 5. ENREGISTREMENT, SUPPRESSION ET IMPORTATION
# =========================================================

def sauvegarder_essai(
    projet, code, jour, age, longueur, largeur,
    hauteur, charge, resistance, forme
):
    try:
        # Vérifier le schéma avant insertion.
        colonnes = {
            c["name"] for c in inspect(ENGINE).get_columns("tests")
        }

        if "project" not in colonnes:
            with ENGINE.begin() as con:
                con.execute(text("""
                    ALTER TABLE tests
                    ADD COLUMN project TEXT NOT NULL DEFAULT 'Projet sans nom'
                """))

        # La valeur project est maintenant explicitement renseignée.
        with ENGINE.begin() as con:
            con.execute(text("""
                INSERT INTO tests (
                    project, sample_code, test_date, age_days,
                    length_mm, width_mm, height_mm,
                    load_kn, strength_mpa, shape
                )
                VALUES (
                    :project, :sample_code, :test_date, :age_days,
                    :length_mm, :width_mm, :height_mm,
                    :load_kn, :strength_mpa, :shape
                )
            """), {
                "project": projet.strip() or "Projet sans nom",
                "sample_code": code.strip(),
                "test_date": jour,
                "age_days": int(age),
                "length_mm": float(longueur),
                "width_mm": float(largeur),
                "height_mm": float(hauteur),
                "load_kn": float(charge),
                "strength_mpa": float(resistance),
                "shape": forme
            })

        enregistrer_action(
            "Ajout d'un essai",
            f"Projet : {projet} ; éprouvette : {code} ; "
            f"résistance : {resistance:.2f} MPa"
        )
        return True

    except Exception as erreur:
        st.error(f"Erreur d'enregistrement : {erreur}")
        return False


def supprimer_essai(identifiant):
    try:
        with ENGINE.begin() as con:
            ligne = con.execute(
                text("SELECT sample_code FROM tests WHERE id = :id"),
                {"id": int(identifiant)}
            ).fetchone()

            if ligne:
                con.execute(
                    text("DELETE FROM tests WHERE id = :id"),
                    {"id": int(identifiant)}
                )

        if ligne:
            enregistrer_action(
                "Suppression d'un essai",
                f"Identifiant {identifiant}, éprouvette {ligne[0]}"
            )

        return ligne is not None

    except Exception as erreur:
        st.error(f"Erreur de suppression : {erreur}")
        return False


def importer_csv(fichier):
    """Importe les essais depuis un CSV exporté par BetonLab."""
    donnees = pd.read_csv(fichier)
    donnees.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in donnees.columns
    ]

    alias = {
        "projet": "project",
        "code": "sample_code",
        "date": "test_date",
        "date_essai": "test_date",
        "age": "age_days",
        "longueur": "length_mm",
        "length": "length_mm",
        "diametre": "length_mm",
        "largeur": "width_mm",
        "width": "width_mm",
        "hauteur": "height_mm",
        "height": "height_mm",
        "charge": "load_kn",
        "charge_kn": "load_kn",
        "resistance": "strength_mpa",
        "resistance_mpa": "strength_mpa",
        "forme": "shape"
    }

    donnees = donnees.rename(columns=alias)

    minimum = {
        "sample_code", "test_date", "age_days",
        "length_mm", "width_mm", "height_mm", "load_kn"
    }

    manquantes = minimum - set(donnees.columns)

    if manquantes:
        raise ValueError(
            "Colonnes nécessaires absentes du CSV : "
            + ", ".join(sorted(manquantes))
        )

    ajoutees = 0

    with ENGINE.begin() as con:
        for _, ligne in donnees.iterrows():
            jour = pd.to_datetime(
                ligne["test_date"], errors="coerce", dayfirst=True
            )

            if pd.isna(jour):
                raise ValueError(
                    f"Date invalide pour l'éprouvette "
                    f"{ligne['sample_code']}."
                )

            forme = str(
                ligne.get("shape", "Prismatique")
            )

            if forme not in ("Prismatique", "Cylindrique"):
                forme = "Prismatique"

            longueur = float(ligne["length_mm"])
            largeur = float(ligne["width_mm"])
            charge = float(ligne["load_kn"])

            if longueur <= 0 or largeur <= 0 or charge <= 0:
                raise ValueError("Dimensions et charge doivent être positives.")

            valeur_resistance = ligne.get("strength_mpa", np.nan)

            if pd.isna(valeur_resistance):
                surface = (
                    np.pi * longueur**2 / 4
                    if forme == "Cylindrique"
                    else longueur * largeur
                )
                valeur_resistance = charge * 1000 / surface

            projet = ligne.get("project", "Projet importé")
            if pd.isna(projet) or not str(projet).strip():
                projet = "Projet importé"

            code = str(ligne["sample_code"]).strip()
            if not code:
                code = "Sans code"

            con.execute(text("""
                INSERT INTO tests (
                    project, sample_code, test_date, age_days,
                    length_mm, width_mm, height_mm,
                    load_kn, strength_mpa, shape
                )
                VALUES (
                    :project, :sample_code, :test_date, :age_days,
                    :length_mm, :width_mm, :height_mm,
                    :load_kn, :strength_mpa, :shape
                )
            """), {
                "project": str(projet),
                "sample_code": code,
                "test_date": jour.date().isoformat(),
                "age_days": int(ligne["age_days"]),
                "length_mm": longueur,
                "width_mm": largeur,
                "height_mm": float(ligne["height_mm"]),
                "load_kn": charge,
                "strength_mpa": float(valeur_resistance),
                "shape": forme
            })

            ajoutees += 1

    enregistrer_action(
        "Importation CSV",
        f"{ajoutees} essai(s) importé(s)"
    )

    return ajoutees


# =========================================================
# 6. CALCUL DE LA RESISTANCE ET STATISTIQUES
# =========================================================

def calculer_resistance(charge_kn, longueur_mm, largeur_mm, forme):
    if charge_kn <= 0 or longueur_mm <= 0 or largeur_mm <= 0:
        raise ValueError("Les dimensions et la charge doivent être positives.")

    surface = (
        np.pi * longueur_mm**2 / 4
        if forme == "Cylindrique"
        else longueur_mm * largeur_mm
    )

    return charge_kn * 1000 / surface


def calculer_statistiques(df):
    x = pd.to_numeric(
        df["strength_mpa"], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return {}

    moyenne = float(np.mean(x))
    ecart_type = float(np.std(x, ddof=1)) if len(x) > 1 else None

    return {
        "nombre": len(x),
        "moyenne": moyenne,
        "mediane": float(np.median(x)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "etendue": float(np.ptp(x)),
        "ecart_type": ecart_type,
        "cv": (
            ecart_type / abs(moyenne) * 100
            if ecart_type is not None and moyenne != 0
            else None
        )
    }


def nombre_francais(valeur, decimales=2):
    if valeur is None or pd.isna(valeur):
        return "—"
    return f"{valeur:,.{decimales}f}".replace(
        ",", " "
    ).replace(".", ",")


# =========================================================
# 7. GRAPHIQUES
# =========================================================

def appliquer_style_graphique(fig, titre):
    c = THEMES[st.session_state.theme_betonlab]

    fig.update_layout(
        title=titre,
        template=c["graphique"],
        font=dict(family="Arial", size=14, color=c["texte"]),
        title_font=dict(size=20, color=c["primaire"]),
        margin=dict(l=25, r=25, t=65, b=35),
        paper_bgcolor=c["fond"],
        plot_bgcolor=c["fond"],
        height=400
    )
    return fig


def graphique_histogramme(df):
    c = THEMES[st.session_state.theme_betonlab]
    x = df["strength_mpa"].dropna()

    fig = px.histogram(
        x=x,
        nbins=max(5, min(20, len(x))),
        color_discrete_sequence=[c["accent"]],
        labels={"value": "Résistance (MPa)", "count": "Nombre d'essais"}
    )

    return appliquer_style_graphique(fig, "Répartition des résistances")


def graphique_evolution(df):
    c = THEMES[st.session_state.theme_betonlab]
    ordre = df.sort_values("test_date").copy()

    fig = px.line(
        ordre,
        x="test_date",
        y="strength_mpa",
        markers=True,
        color_discrete_sequence=[c["accent"]],
        labels={
            "test_date": "Date de l'essai",
            "strength_mpa": "Résistance (MPa)"
        },
        hover_data=["sample_code", "age_days", "project"]
    )

    return appliquer_style_graphique(fig, "Évolution de la résistance")


def graphique_boite(df):
    c = THEMES[st.session_state.theme_betonlab]

    fig = px.box(
        df,
        y="strength_mpa",
        points="all",
        color_discrete_sequence=[c["accent"]],
        labels={"strength_mpa": "Résistance (MPa)"}
    )

    return appliquer_style_graphique(fig, "Dispersion des résistances")


def graphique_normale(df):
    c = THEMES[st.session_state.theme_betonlab]
    x = df["strength_mpa"].dropna().to_numpy(dtype=float)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=x,
        histnorm="probability density",
        nbinsx=max(5, min(20, len(x))),
        name="Résultats observés",
        marker_color=c["primaire"],
        opacity=0.6
    ))

    if len(x) >= 2:
        moyenne = float(np.mean(x))
        ecart_type = float(np.std(x, ddof=1))

        if ecart_type > 0:
            xx = np.linspace(
                float(np.min(x)) - ecart_type,
                float(np.max(x)) + ecart_type,
                300
            )

            fig.add_trace(go.Scatter(
                x=xx,
                y=stats.norm.pdf(xx, moyenne, ecart_type),
                mode="lines",
                name="Modèle normal",
                line=dict(color=c["accent"], width=3)
            ))

    fig.update_layout(
        xaxis_title="Résistance (MPa)",
        yaxis_title="Densité de probabilité"
    )

    return appliquer_style_graphique(fig, "Distribution normale estimée")


def afficher_carte(titre, valeur, unite=""):
    c = THEMES[st.session_state.theme_betonlab]

    st.markdown(
        f"""
        <div style="
            background:{c['panneau']};
            border:1px solid {c['bordure']};
            border-radius:12px;
            padding:16px;
            min-height:92px;">
            <div style="color:{c['secondaire']};font-size:14px;">
                {titre}
            </div>
            <div style="color:{c['texte']};font-size:24px;font-weight:700;">
                {valeur} {unite}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 8. APPARENCE, STOCKAGE ET SELECTION DES DONNEES
# =========================================================

with st.sidebar:
    st.title("🏗️ BetonLab")

    st.radio(
        "Apparence",
        ["Clair", "Sombre"],
        key="theme_betonlab"
    )

    st.divider()

    if TYPE_BASE == "postgresql":
        st.success("Stockage PostgreSQL distant activé")
    else:
        st.warning(
            "Stockage SQLite local. Sur Streamlit Cloud, les données "
            "ne sont pas garanties après un redémarrage. Configure "
            "DATABASE_URL pour utiliser PostgreSQL."
        )

    df_complet = charger_essais()

    projets = (
        sorted(df_complet["project"].dropna().unique().tolist())
        if not df_complet.empty
        else []
    )

    choix_projet = st.selectbox(
        "Données à analyser",
        ["Tous les projets"] + projets,
        help="Choisis un projet existant pour analyser uniquement ses essais."
    )

    st.caption(
        "Le choix du projet filtre les statistiques et les graphiques, "
        "mais ne supprime aucune donnée."
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Tableau de bord",
            "Enregistrer un essai",
            "Base des essais",
            "Statistiques",
            "Probabilités",
            "Diagrammes",
            "Historique d'utilisation"
        ]
    )

    st.caption("BetonLab • Version 3.0")


C = THEMES[st.session_state.theme_betonlab]

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color:{C["fond"]};
        color:{C["texte"]};
    }}
    [data-testid="stSidebar"] {{
        background-color:{C["panneau"]};
        border-right:1px solid {C["bordure"]};
    }}
    [data-testid="stSidebar"] * {{
        color:{C["texte"]};
    }}
    h1,h2,h3,h4,p,label {{
        color:{C["texte"]};
    }}
    [data-testid="stMetric"] {{
        background:{C["panneau"]};
        border:1px solid {C["bordure"]};
        padding:15px;
        border-radius:12px;
    }}
    .hero {{
        background:linear-gradient(110deg,{C["primaire"]},{C["accent"]});
        padding:24px;
        border-radius:16px;
        margin-bottom:20px;
    }}
    .hero h1,.hero p {{
        color:white !important;
    }}
    .hero h1 {{
        margin:0;
        font-size:2.1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

df = df_complet.copy()

if choix_projet != "Tous les projets":
    df = df[df["project"] == choix_projet]

if st.session_state.get("derniere_page") != page:
    st.session_state.derniere_page = page
    enregistrer_action("Consultation d'une rubrique", page)

st.markdown("""
<div class="hero">
    <h1>🏗️ BetonLab</h1>
    <p>Analyse des essais de résistance à la compression du béton</p>
    <p>Statistiques • Probabilités • Diagrammes • Données enregistrées</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 9. TABLEAU DE BORD
# =========================================================

if page == "Tableau de bord":
    st.header("📊 Tableau de bord")

    if df.empty:
        st.info(
            "Aucun essai ne correspond au projet choisi. "
            "Enregistre un essai ou sélectionne un autre projet."
        )
    else:
        s = calculer_statistiques(df)

        colonnes = st.columns(4)
        valeurs = [
            ("Nombre d'essais", str(s["nombre"])),
            ("Résistance moyenne", f'{s["moyenne"]:.2f} MPa'),
            ("Résistance maximale", f'{s["maximum"]:.2f} MPa'),
            ("Résistance minimale", f'{s["minimum"]:.2f} MPa')
        ]

        for col, (titre, valeur) in zip(colonnes, valeurs):
            with col:
                st.metric(titre, valeur)

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(
                graphique_histogramme(df),
                use_container_width=True
            )

        with c2:
            if len(df) >= 2:
                st.plotly_chart(
                    graphique_evolution(df),
                    use_container_width=True
                )
            else:
                st.info("Ajoute plusieurs essais pour afficher l'évolution.")

        st.subheader("Essais récents")

        recents = df.head(10).copy()
        recents["test_date"] = recents["test_date"].dt.strftime("%d/%m/%Y")

        st.dataframe(
            recents[[
                "project", "sample_code", "test_date",
                "age_days", "load_kn", "strength_mpa", "shape"
            ]].rename(columns={
                "project": "Projet",
                "sample_code": "Éprouvette",
                "test_date": "Date",
                "age_days": "Âge (jours)",
                "load_kn": "Charge (kN)",
                "strength_mpa": "Résistance (MPa)",
                "shape": "Forme"
            }),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 10. ENREGISTRER UN ESSAI
# =========================================================

elif page == "Enregistrer un essai":
    st.header("🧪 Enregistrer un essai de compression")

    with st.form("formulaire_essai", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            projet = st.text_input(
                "Nom du projet *",
                value=(
                    choix_projet
                    if choix_projet != "Tous les projets"
                    else "Projet général"
                ),
                help="Les essais sont classés par projet."
            )

            code = st.text_input(
                "Code de l'éprouvette *",
                placeholder="Ex. BET-001"
            )

            jour = st.date_input(
                "Date de l'essai",
                value=date.today(),
                format="DD/MM/YYYY"
            )

            age = st.number_input(
                "Âge du béton (jours)",
                min_value=1,
                max_value=36500,
                value=28
            )

            forme = st.selectbox(
                "Forme de l'éprouvette",
                ["Prismatique", "Cylindrique"]
            )

        with c2:
            if forme == "Cylindrique":
                longueur = st.number_input(
                    "Diamètre (mm)",
                    min_value=0.1,
                    value=150.0
                )
                largeur = longueur
                st.caption("Surface : π × diamètre² / 4.")
            else:
                longueur = st.number_input(
                    "Longueur de la surface chargée (mm)",
                    min_value=0.1,
                    value=150.0
                )
                largeur = st.number_input(
                    "Largeur de la surface chargée (mm)",
                    min_value=0.1,
                    value=150.0
                )

            hauteur = st.number_input(
                "Hauteur (mm)",
                min_value=0.1,
                value=300.0
            )

            charge = st.number_input(
                "Charge de rupture (kN)",
                min_value=0.1,
                value=650.0
            )

        envoyer = st.form_submit_button(
            "Calculer et enregistrer",
            use_container_width=True
        )

    if envoyer:
        if not projet.strip():
            st.error("Saisis le nom du projet.")
        elif not code.strip():
            st.error("Saisis le code de l'éprouvette.")
        else:
            try:
                resistance = calculer_resistance(
                    charge, longueur, largeur, forme
                )

                succes = sauvegarder_essai(
                    projet.strip(),
                    code.strip(),
                    jour.isoformat(),
                    int(age),
                    float(longueur),
                    float(largeur),
                    float(hauteur),
                    float(charge),
                    float(resistance),
                    forme
                )

                if succes:
                    st.success("Essai enregistré avec succès.")
                    st.metric(
                        "Résistance calculée",
                        f"{resistance:.2f} MPa"
                    )
                    st.rerun()

            except ValueError as erreur:
                st.error(str(erreur))


# =========================================================
# 11. BASE DES ESSAIS, IMPORT ET EXPORT
# =========================================================

elif page == "Base des essais":
    st.header("🗂️ Base des essais enregistrés")

    st.write(
        "Cette page affiche les essais de tous les projets. "
        "Tu peux les rechercher, les exporter, importer un CSV "
        "ou supprimer un essai."
    )

    if df_complet.empty:
        st.info("Aucun essai n'est encore enregistré.")
    else:
        recherche = st.text_input(
            "Recherche par projet ou code d'éprouvette"
        )

        vue = df_complet.copy()

        if recherche.strip():
            masque = (
                vue["project"].astype(str).str.contains(
                    recherche, case=False, na=False
                )
                | vue["sample_code"].astype(str).str.contains(
                    recherche, case=False, na=False
                )
            )
            vue = vue[masque]

        affichage = vue.copy()
        affichage["test_date"] = affichage["test_date"].dt.strftime("%d/%m/%Y")

        st.dataframe(
            affichage.rename(columns={
                "id": "Identifiant",
                "project": "Projet",
                "sample_code": "Éprouvette",
                "test_date": "Date",
                "age_days": "Âge (jours)",
                "length_mm": "Longueur ou diamètre (mm)",
                "width_mm": "Largeur (mm)",
                "height_mm": "Hauteur (mm)",
                "load_kn": "Charge (kN)",
                "strength_mpa": "Résistance (MPa)",
                "shape": "Forme"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Exporter les données")

    st.download_button(
        "⬇️ Télécharger tous les essais (CSV)",
        data=df_complet.to_csv(index=False).encode("utf-8-sig"),
        file_name="betonlab_essais.csv",
        mime="text/csv"
    )

    st.caption(
        "Conserve ce fichier CSV comme copie de sécurité. "
        "Il peut être réimporté si nécessaire."
    )

    st.divider()
    st.subheader("Importer des données déjà enregistrées")

    fichier = st.file_uploader(
        "Choisir un fichier CSV issu de BetonLab",
        type=["csv"]
    )

    if fichier is not None:
        st.caption("Aperçu du fichier à importer")
        st.dataframe(
            pd.read_csv(fichier).head(10),
            use_container_width=True,
            hide_index=True
        )
        fichier.seek(0)

        if st.button("Importer ces essais dans la base"):
            try:
                fichier.seek(0)
                n = importer_csv(fichier)
                st.success(f"{n} essai(s) importé(s) avec succès.")
                st.rerun()
            except Exception as erreur:
                st.error(f"Importation impossible : {erreur}")

    st.divider()
    st.subheader("Supprimer un essai")

    ids = df_complet["id"].astype(int).tolist()

    if ids:
        identifiant = st.selectbox(
            "Choisir l'identifiant",
            ids,
            format_func=lambda i: (
                f"ID {i} — "
                f"{df_complet.loc[df_complet['id'] == i, 'sample_code'].iloc[0]}"
            )
        )

        confirmer = st.checkbox(
            "Je confirme la suppression définitive de cet essai."
        )

        if st.button("Supprimer l'essai"):
            if confirmer:
                if supprimer_essai(int(identifiant)):
                    st.success("Essai supprimé.")
                    st.rerun()
            else:
                st.warning("Confirme la suppression avant de continuer.")


# =========================================================
# 12. STATISTIQUES
# =========================================================

elif page == "Statistiques":
    st.header("📊 Analyse statistique")

    s = calculer_statistiques(df)

    if not s:
        st.info("Aucune donnée disponible pour le projet sélectionné.")
    else:
        indicateurs = [
            ("Essais", str(s["nombre"])),
            ("Moyenne", f'{s["moyenne"]:.2f} MPa'),
            ("Médiane", f'{s["mediane"]:.2f} MPa'),
            ("Étendue", f'{s["etendue"]:.2f} MPa'),
            ("Minimum", f'{s["minimum"]:.2f} MPa'),
            ("Maximum", f'{s["maximum"]:.2f} MPa'),
            (
                "Écart-type",
                f'{s["ecart_type"]:.2f} MPa'
                if s["ecart_type"] is not None else "—"
            ),
            (
                "Coefficient de variation",
                f'{s["cv"]:.2f}%'
                if s["cv"] is not None else "—"
            )
        ]

        for debut in range(0, len(indicateurs), 4):
            colonnes = st.columns(4)
            for col, (titre, valeur) in zip(
                colonnes, indicateurs[debut:debut + 4]
            ):
                col.metric(titre, valeur)

        st.plotly_chart(
            graphique_boite(df),
            use_container_width=True
        )

        valeurs = df["strength_mpa"].dropna().to_numpy(dtype=float)

        st.subheader("Test de normalité de Shapiro-Wilk")

        if len(valeurs) >= 3 and np.std(valeurs) > 0:
            echantillon = (
                valeurs if len(valeurs) <= 5000
                else np.random.default_rng(42).choice(
                    valeurs, size=5000, replace=False
                )
            )

            resultat = stats.shapiro(echantillon)

            st.write(f"Statistique : **{resultat.statistic:.4f}**")
            st.write(f"Valeur p : **{resultat.pvalue:.4f}**")

            if resultat.pvalue < 0.05:
                st.warning(
                    "Le test fournit des éléments contre l'hypothèse "
                    "de normalité au seuil de 5 %."
                )
            else:
                st.info(
                    "Le test ne permet pas de rejeter la normalité au "
                    "seuil de 5 %. Cela ne prouve pas que les données "
                    "sont normalement distribuées."
                )
        else:
            st.info("Il faut au moins trois valeurs avec une dispersion non nulle.")


# =========================================================
# 13. PROBABILITES
# =========================================================

elif page == "Probabilités":
    st.header("🎯 Analyse des probabilités")

    valeurs = df["strength_mpa"].dropna().to_numpy(dtype=float)

    if len(valeurs) < 2:
        st.warning(
            "Il faut au moins deux essais pour ajuster un modèle normal."
        )
    else:
        moyenne = float(np.mean(valeurs))
        ecart_type = float(np.std(valeurs, ddof=1))

        c1, c2 = st.columns(2)
        c1.metric("Moyenne", f"{moyenne:.2f} MPa")
        c2.metric("Écart-type", f"{ecart_type:.2f} MPa")

        cible = st.number_input(
            "Résistance cible (MPa)",
            min_value=0.0,
            value=float(round(moyenne, 1)),
            step=0.5
        )

        if ecart_type == 0:
            st.warning("L'écart-type est nul : le modèle normal ne peut pas être estimé.")
        else:
            p_inf = float(
                stats.norm.cdf(cible, loc=moyenne, scale=ecart_type)
            )
            p_sup = 1 - p_inf

            c1, c2 = st.columns(2)
            c1.metric(f"P(X ≤ {cible:.2f})", f"{p_inf * 100:.2f}%")
            c2.metric(f"P(X ≥ {cible:.2f})", f"{p_sup * 100:.2f}%")

            couleurs = THEMES[st.session_state.theme_betonlab]
            xx = np.linspace(
                min(0, moyenne - 4 * ecart_type),
                max(moyenne + 4 * ecart_type, cible + ecart_type),
                400
            )
            yy = stats.norm.pdf(xx, moyenne, ecart_type)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xx, y=yy, mode="lines",
                name="Densité normale",
                line=dict(color=couleurs["primaire"], width=3)
            ))

            masque = xx <= cible
            fig.add_trace(go.Scatter(
                x=xx[masque], y=yy[masque],
                mode="lines", fill="tozeroy",
                name="Sous la cible",
                line=dict(width=0),
                fillcolor="rgba(34,166,153,0.35)"
            ))

            fig.add_vline(
                x=cible,
                line_dash="dash",
                line_color=couleurs["danger"],
                annotation_text="Cible"
            )

            st.plotly_chart(
                appliquer_style_graphique(
                    fig, "Probabilité selon le modèle normal"
                ),
                use_container_width=True
            )

            st.caption(
                "Les probabilités sont des estimations statistiques. "
                "Elles ne constituent pas à elles seules un contrôle "
                "de conformité du béton."
            )


# =========================================================
# 14. DIAGRAMMES
# =========================================================

elif page == "Diagrammes":
    st.header("📈 Galerie de diagrammes")

    if df.empty:
        st.info("Aucune donnée disponible.")
    else:
        choix = st.selectbox(
            "Type de diagramme",
            [
                "Histogramme",
                "Évolution dans le temps",
                "Boîte à moustaches",
                "Distribution normale",
                "Résistance en fonction de l'âge"
            ]
        )

        if choix == "Histogramme":
            fig = graphique_histogramme(df)

        elif choix == "Évolution dans le temps":
            fig = graphique_evolution(df)

        elif choix == "Boîte à moustaches":
            fig = graphique_boite(df)

        elif choix == "Distribution normale":
            fig = graphique_normale(df)

        else:
            fig = px.scatter(
                df,
                x="age_days",
                y="strength_mpa",
                color="project",
                hover_data=["sample_code"],
                labels={
                    "age_days": "Âge du béton (jours)",
                    "strength_mpa": "Résistance (MPa)",
                    "project": "Projet"
                }
            )
            fig = appliquer_style_graphique(
                fig, "Résistance en fonction de l'âge"
            )

        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 15. HISTORIQUE D'UTILISATION
# =========================================================

elif page == "Historique d'utilisation":
    st.header("🕒 Historique d'utilisation")

    historique = charger_historique()

    c1, c2 = st.columns(2)
    c1.metric("Actions enregistrées", len(historique))
    c2.metric("Essais dans la base", len(df_complet))

    if historique.empty:
        st.info("Aucune action enregistrée pour l'instant.")
    else:
        actions = sorted(
            historique["Action"].dropna().unique().tolist()
        )

        filtre = st.selectbox(
            "Filtrer les actions",
            ["Toutes les actions"] + actions
        )

        vue = historique.copy()

        if filtre != "Toutes les actions":
            vue = vue[vue["Action"] == filtre]

        st.dataframe(
            vue,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Exporter l'historique",
            data=vue.to_csv(index=False).encode("utf-8-sig"),
            file_name="betonlab_historique.csv",
            mime="text/csv"
        )


# =========================================================
# 16. PIED DE PAGE
# =========================================================

st.divider()

st.caption(
    f"BetonLab • Résistance du béton en MPa • "
    f"Thème {st.session_state.theme_betonlab} • "
    f"Stockage : {TYPE_BASE}"
)
