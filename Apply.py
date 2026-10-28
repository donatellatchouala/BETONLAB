import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="BetonControl",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "betonlab.db")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
try:
    if not DATABASE_URL and "DATABASE_URL" in st.secrets:
        DATABASE_URL = str(st.secrets["DATABASE_URL"]).strip()
except Exception:
    pass

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

try:
    if DATABASE_URL:
        moteur = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        MODE_STOCKAGE = "PostgreSQL"
    else:
        moteur = create_engine(
            "sqlite:///" + DB_PATH.replace("\\", "/"),
            connect_args={"timeout": 20},
        )
        MODE_STOCKAGE = "SQLite"
except Exception as exc:
    st.error("Impossible de préparer la base de données. Vérifiez DATABASE_URL et les dépendances.")
    st.exception(exc)
    st.stop()

# =========================================================
# 2. LANGUES
# =========================================================
TEXTES = {
    "Français": {
        "title": "Analyse des essais de résistance du béton",
        "dashboard": "Tableau de bord",
        "add": "Enregistrer un essai",
        "database": "Base des essais",
        "statistics": "Statistiques",
        "probability": "Probabilités",
        "charts": "Diagrammes",
        "history": "Historique",
        "language": "Langue",
        "appearance": "Apparence",
        "light": "Clair",
        "dark": "Sombre",
        "overview": "Vue d'ensemble du laboratoire",
        "no_data": "Aucun essai enregistré.",
        "tests": "Nombre d'essais",
        "mean": "Résistance moyenne",
        "minimum": "Résistance minimale",
        "maximum": "Résistance maximale",
        "code": "Code de l'éprouvette",
        "test_date": "Date de l'essai",
        "age": "Âge du béton (jours)",
        "shape": "Forme de l'éprouvette",
        "prismatic": "Prismatique",
        "cylindrical": "Cylindrique",
        "length": "Longueur de la surface chargée (mm)",
        "width": "Largeur de la surface chargée (mm)",
        "diameter": "Diamètre (mm)",
        "height": "Hauteur (mm)",
        "load": "Charge maximale de rupture (kN)",
        "save": "Calculer et enregistrer",
        "success": "Essai enregistré avec succès.",
        "empty_code": "Veuillez renseigner le code de l'éprouvette.",
        "strength": "Résistance calculée",
        "search": "Rechercher une éprouvette",
        "export": "Exporter en CSV",
        "import": "Importer des essais depuis un CSV",
        "import_button": "Importer le fichier",
        "demo": "Ajouter des données de démonstration",
        "demo_added": "Données de démonstration ajoutées.",
        "demo_exists": "Des données sont déjà enregistrées.",
        "target": "Résistance cible (MPa)",
        "below": "Probabilité d'être inférieur ou égal à la cible",
        "above": "Probabilité d'être supérieur à la cible",
        "actions": "Actions enregistrées",
        "delete": "Supprimer un essai",
        "confirm": "Je confirme la suppression",
        "deleted": "Essai supprimé.",
        "error": "Erreur",
        "storage_local": "Stockage SQLite local.",
        "storage_cloud": "Stockage PostgreSQL configuré.",
        "persistent_warning": "Sur Streamlit Cloud, SQLite local peut disparaître lors d'un redéploiement. Configurez DATABASE_URL avec une base PostgreSQL persistante.",
        "import_success": "Importation terminée.",
        "no_results": "Aucun résultat à afficher.",
        "project": "Projet",
        "project_name": "Nom du projet",
        "normality": "Test de normalité",
        "history_empty": "Aucune opération enregistrée.",
        "chart_distribution": "Distribution des résistances",
        "chart_evolution": "Évolution des résistances",
        "chart_box": "Dispersion des résistances",
        "reload_data": "Recharger les données",
        "saved_data": "Afficher les données enregistrées",
        "saved_data_title": "Données déjà enregistrées",
        "backup_csv": "Télécharger une sauvegarde CSV",
        "no_saved_data": "Aucune donnée enregistrée dans la base.",
        "demo_intro": "Ces données fictives servent à tester les calculs et les graphiques.",
        "missing_columns": "Colonnes manquantes",
        "invalid_id": "Identifiant introuvable.",
        "delete_id": "ID de l'essai à supprimer",
        "chart_type": "Type de diagramme",
        "histogram": "Histogramme",
        "evolution": "Évolution des résistances",
        "boxplot": "Boîte à moustaches",
        "age_chart": "Résistance en fonction de l'âge",
        "normality_warning": "Le test apporte des éléments contre l'hypothèse de normalité.",
        "normality_ok": "Le test ne permet pas de rejeter l'hypothèse de normalité.",
        "need_two": "Enregistrez au moins deux essais.",
        "std_zero": "L'écart-type est nul.",
        "probability_note": "Ces probabilités sont des estimations statistiques et ne remplacent pas les critères réglementaires de conformité du béton.",
        "saved_data_help": "Les données proviennent de la base active configurée pour cette application.",
        "footer": "Statistiques • Probabilités • Diagrammes • Données",
    },
    "English": {
        "title": "Concrete Compression Test Analyzer",
        "dashboard": "Dashboard",
        "add": "Record a test",
        "database": "Test database",
        "statistics": "Statistics",
        "probability": "Probability",
        "charts": "Charts",
        "history": "History",
        "language": "Language",
        "appearance": "Appearance",
        "light": "Light",
        "dark": "Dark",
        "overview": "Laboratory overview",
        "no_data": "No tests recorded.",
        "tests": "Number of tests",
        "mean": "Mean strength",
        "minimum": "Minimum strength",
        "maximum": "Maximum strength",
        "code": "Specimen code",
        "test_date": "Test date",
        "age": "Concrete age (days)",
        "shape": "Specimen shape",
        "prismatic": "Prismatic",
        "cylindrical": "Cylindrical",
        "length": "Loaded-face length (mm)",
        "width": "Loaded-face width (mm)",
        "diameter": "Diameter (mm)",
        "height": "Specimen height (mm)",
        "load": "Maximum failure load (kN)",
        "save": "Calculate and save",
        "success": "Test saved successfully.",
        "empty_code": "Please enter a specimen code.",
        "strength": "Calculated strength",
        "search": "Search specimens",
        "export": "Export CSV",
        "import": "Import tests from CSV",
        "import_button": "Import file",
        "demo": "Add demonstration data",
        "demo_added": "Demonstration data added.",
        "demo_exists": "Data is already stored.",
        "target": "Target strength (MPa)",
        "below": "Probability of being below or equal to target",
        "above": "Probability of being above target",
        "actions": "Recorded actions",
        "delete": "Delete a test",
        "confirm": "I confirm deletion",
        "deleted": "Test deleted.",
        "error": "Error",
        "storage_local": "Local SQLite storage.",
        "storage_cloud": "PostgreSQL storage configured.",
        "persistent_warning": "On Streamlit Cloud, local SQLite may disappear after redeployment. Configure DATABASE_URL with a persistent PostgreSQL database.",
        "import_success": "Import completed.",
        "no_results": "No results to display.",
        "project": "Project",
        "project_name": "Project name",
        "normality": "Normality test",
        "history_empty": "No recorded actions.",
        "chart_distribution": "Strength distribution",
        "chart_evolution": "Strength evolution",
        "chart_box": "Strength dispersion",
        "reload_data": "Reload data",
        "saved_data": "View saved data",
        "saved_data_title": "Previously saved data",
        "backup_csv": "Download CSV backup",
        "no_saved_data": "No data stored in the database.",
        "demo_intro": "These fictional records help test calculations and charts.",
        "missing_columns": "Missing columns",
        "invalid_id": "ID not found.",
        "delete_id": "Test ID to delete",
        "chart_type": "Chart type",
        "histogram": "Histogram",
        "evolution": "Strength evolution",
        "boxplot": "Box plot",
        "age_chart": "Strength by concrete age",
        "normality_warning": "The test provides evidence against the normality assumption.",
        "normality_ok": "The test does not reject the normality assumption.",
        "need_two": "Record at least two tests.",
        "std_zero": "The standard deviation is zero.",
        "probability_note": "These probabilities are statistical estimates and do not replace regulatory concrete compliance criteria.",
        "saved_data_help": "Data is loaded from the active database configured for this application.",
        "footer": "Statistics • Probability • Charts • Data",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "Français"
if "theme" not in st.session_state:
    st.session_state.theme = "Clair"

def T(key):
    return TEXTES[st.session_state.lang].get(key, key)

# =========================================================
# 3. BASE DE DONNÉES ET MIGRATIONS
# =========================================================
def colonnes_table(nom):
    insp = inspect(moteur)
    if not insp.has_table(nom):
        return []
    return insp.get_columns(nom)

def initialiser_base():
    id_type = "SERIAL PRIMARY KEY" if moteur.dialect.name == "postgresql" else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with moteur.begin() as con:
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS tests (
                id {id_type},
                sample_code TEXT NOT NULL,
                test_date TEXT NOT NULL,
                age_days INTEGER NOT NULL DEFAULT 28,
                length_mm REAL NOT NULL DEFAULT 150,
                width_mm REAL NOT NULL DEFAULT 150,
                height_mm REAL NOT NULL DEFAULT 300,
                load_kn REAL NOT NULL DEFAULT 1,
                strength_mpa REAL NOT NULL DEFAULT 0,
                shape TEXT DEFAULT 'Prismatique',
                project TEXT DEFAULT 'Projet général',
                specimen_reference TEXT DEFAULT ''
            )
        """))
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS historique (
                id {id_type},
                date_heure TEXT NOT NULL DEFAULT '',
                action TEXT NOT NULL DEFAULT '',
                details TEXT DEFAULT ''
            )
        """))

    # Migration non destructive : ajouter les champs manquants.
    colonnes = {c["name"] for c in colonnes_table("tests")}
    types = {
        "sample_code": "TEXT DEFAULT ''",
        "test_date": "TEXT DEFAULT ''",
        "age_days": "INTEGER DEFAULT 28",
        "length_mm": "REAL DEFAULT 150",
        "width_mm": "REAL DEFAULT 150",
        "height_mm": "REAL DEFAULT 300",
        "load_kn": "REAL DEFAULT 1",
        "strength_mpa": "REAL DEFAULT 0",
        "shape": "TEXT DEFAULT 'Prismatique'",
        "project": "TEXT DEFAULT 'Projet général'",
        "specimen_reference": "TEXT DEFAULT ''",
    }
    for nom, definition in types.items():
        if nom not in colonnes:
            with moteur.begin() as con:
                con.execute(text(f'ALTER TABLE tests ADD COLUMN "{nom}" {definition}'))

    colonnes_hist = {c["name"] for c in colonnes_table("historique")}
    for nom, definition in {
        "date_heure": "TEXT DEFAULT ''",
        "action": "TEXT DEFAULT ''",
        "details": "TEXT DEFAULT ''",
    }.items():
        if nom not in colonnes_hist:
            with moteur.begin() as con:
                con.execute(text(f'ALTER TABLE historique ADD COLUMN "{nom}" {definition}'))

def enregistrer_action(action, details=""):
    try:
        with moteur.begin() as con:
            con.execute(text("""
                INSERT INTO historique (date_heure, action, details)
                VALUES (:date_heure, :action, :details)
            """), {
                "date_heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": action,
                "details": details,
            })
    except Exception:
        # L'historique ne doit pas bloquer un enregistrement d'essai.
        pass

def charger_essais():
    with moteur.connect() as con:
        df = pd.read_sql(text("SELECT * FROM tests ORDER BY id DESC"), con)

    correspondances = {
        "project_name": "project",
        "specimen_ref": "specimen_reference",
        "reference": "specimen_reference",
        "specimen": "specimen_reference",
        "resistance": "strength_mpa",
        "strength": "strength_mpa",
        "charge": "load_kn",
        "age": "age_days",
        "forme": "shape",
    }
    for ancien, nouveau in correspondances.items():
        if ancien in df.columns and nouveau not in df.columns:
            df[nouveau] = df[ancien]

    for col in [
        "id", "sample_code", "test_date", "age_days", "length_mm",
        "width_mm", "height_mm", "load_kn", "strength_mpa", "shape",
        "project", "specimen_reference",
    ]:
        if col not in df.columns:
            df[col] = None

    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    for col in ["age_days", "length_mm", "width_mm", "height_mm", "load_kn", "strength_mpa"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def charger_historique():
    with moteur.connect() as con:
        return pd.read_sql(text("""
            SELECT date_heure AS "Date / Time",
                   action AS "Action",
                   details AS "Details"
            FROM historique
            ORDER BY id DESC
        """), con)

def inserer_essai(donnees):
    """Insère un essai et génère une référence distincte en cas de doublon."""
    colonnes = colonnes_table("tests")
    noms = {c["name"] for c in colonnes}
    code = str(donnees.get("sample_code", "")).strip()
    if not code:
        raise ValueError(T("empty_code"))

    ref_base = str(donnees.get("specimen_reference") or code).strip() or code

    # Le suffixe est ajouté à la référence et non au code de l'éprouvette.
    with moteur.connect() as con:
        refs = set()
        if "specimen_reference" in noms:
            rows = con.execute(text("""
                SELECT specimen_reference FROM tests
                WHERE specimen_reference IS NOT NULL
            """))
            refs = {str(r[0]) for r in rows}

    reference = ref_base
    n = 2
    while reference in refs:
        reference = f"{ref_base}-{n}"
        n += 1

    valeurs = {
        "sample_code": code,
        "test_date": str(donnees["test_date"]),
        "age_days": int(donnees["age_days"]),
        "length_mm": float(donnees["length_mm"]),
        "width_mm": float(donnees["width_mm"]),
        "height_mm": float(donnees["height_mm"]),
        "load_kn": float(donnees["load_kn"]),
        "strength_mpa": float(donnees["strength_mpa"]),
        "shape": str(donnees["shape"]),
        "project": str(donnees["project"]),
        "project_name": str(donnees["project"]),
        "specimen_reference": reference,
        "specimen_ref": reference,
        "reference": reference,
        "specimen": reference,
        "resistance": float(donnees["strength_mpa"]),
        "strength": float(donnees["strength_mpa"]),
        "charge": float(donnees["load_kn"]),
        "age": int(donnees["age_days"]),
        "forme": str(donnees["shape"]),
    }

    # Remplir les autres colonnes NOT NULL héritées d'anciennes versions.
    for c in colonnes:
        nom = c["name"]
        if nom in valeurs or nom == "id":
            continue
        if c.get("nullable", True) is False and c.get("default") is None:
            low = nom.lower()
            if "project" in low or "projet" in low:
                valeurs[nom] = donnees["project"]
            elif "specimen" in low or "reference" in low:
                valeurs[nom] = reference
            elif "date" in low:
                valeurs[nom] = str(donnees["test_date"])
            elif "age" in low:
                valeurs[nom] = int(donnees["age_days"])
            elif "shape" in low or "forme" in low:
                valeurs[nom] = str(donnees["shape"])
            elif "load" in low or "charge" in low:
                valeurs[nom] = float(donnees["load_kn"])
            elif "strength" in low or "resistance" in low:
                valeurs[nom] = float(donnees["strength_mpa"])
            elif "code" in low:
                valeurs[nom] = code
            elif "length" in low or "longueur" in low or "diameter" in low:
                valeurs[nom] = float(donnees["length_mm"])
            elif "width" in low or "largeur" in low:
                valeurs[nom] = float(donnees["width_mm"])
            elif "height" in low or "hauteur" in low:
                valeurs[nom] = float(donnees["height_mm"])
            else:
                valeurs[nom] = ""

    valeurs = {k: v for k, v in valeurs.items() if k in noms and k != "id"}
    if not valeurs:
        raise RuntimeError("Aucune colonne compatible pour l'insertion.")

    cols_sql = ", ".join(f'"{k}"' for k in valeurs)
    params_sql = ", ".join(f":p{i}" for i in range(len(valeurs)))
    stmt = text(f"INSERT INTO tests ({cols_sql}) VALUES ({params_sql})")
    params = {f"p{i}": v for i, v in enumerate(valeurs.values())}

    try:
        with moteur.begin() as con:
            con.execute(stmt, params)
    except IntegrityError:
        # Une autre colonne UNIQUE peut exister dans une ancienne base.
        # On fournit un message clair sans effacer ni écraser les données.
        raise ValueError(
            "La base contient une autre contrainte UNIQUE (par exemple sur le code). "
            "Aucune donnée existante n'a été supprimée. Vérifiez les contraintes "
            "de l'ancienne base ou choisissez un code d'éprouvette différent."
        )
    return reference

initialiser_base()

# =========================================================
# 4. CALCULS
# =========================================================
def calculer_resistance(charge_kn, dimension1_mm, dimension2_mm, forme):
    if charge_kn <= 0:
        raise ValueError("La charge doit être positive.")
    if dimension1_mm <= 0 or dimension2_mm <= 0:
        raise ValueError("Les dimensions doivent être positives.")
    surface = (
        np.pi * dimension1_mm**2 / 4
        if forme == "Cylindrique"
        else dimension1_mm * dimension2_mm
    )
    return float(charge_kn * 1000 / surface)  # N/mm² = MPa

def calculer_statistiques(df):
    x = pd.to_numeric(df["strength_mpa"], errors="coerce").dropna().to_numpy(dtype=float)
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
        "cv": 100 * ecart_type / abs(moyenne) if ecart_type is not None and moyenne != 0 else None,
    }

def ajouter_donnees_demo():
    if not charger_essais().empty:
        return False
    exemples = [
        ("BET-001", 7, 150, 150, 300, 420, "Prismatique"),
        ("BET-002", 7, 150, 150, 300, 450, "Prismatique"),
        ("BET-003", 14, 150, 150, 300, 520, "Prismatique"),
        ("BET-004", 14, 150, 150, 300, 560, "Prismatique"),
        ("BET-005", 28, 150, 150, 300, 650, "Prismatique"),
        ("BET-006", 28, 150, 150, 300, 680, "Prismatique"),
        ("BET-007", 28, 150, 150, 300, 710, "Prismatique"),
        ("BET-008", 28, 150, 150, 300, 735, "Prismatique"),
        ("BET-009", 56, 150, 150, 300, 760, "Prismatique"),
        ("BET-010", 56, 150, 150, 300, 790, "Prismatique"),
    ]
    for code, age, longueur, largeur, hauteur, charge, forme in exemples:
        resistance = calculer_resistance(charge, longueur, largeur, forme)
        inserer_essai({
            "sample_code": code,
            "test_date": date.today().isoformat(),
            "age_days": age,
            "length_mm": longueur,
            "width_mm": largeur,
            "height_mm": hauteur,
            "load_kn": charge,
            "strength_mpa": resistance,
            "shape": forme,
            "project": "Projet de démonstration",
            "specimen_reference": code,
        })
    enregistrer_action("Ajout des données de démonstration", f"{len(exemples)} essais")
    return True

# =========================================================
# 5. THÈME ET GRAPHIQUES
# =========================================================
THEMES = {
    "Clair": {"fond": "#FFFFFF", "panneau": "#F2F5FA", "texte": "#172B4D", "primaire": "#173B67", "accent": "#087F8C", "bordure": "#D9E2EF", "graphique": "plotly_white"},
    "Sombre": {"fond": "#101114", "panneau": "#202329", "texte": "#FFFFFF", "primaire": "#8CB8FF", "accent": "#38D6C5", "bordure": "#444954", "graphique": "plotly_dark"},
}
C = THEMES[st.session_state.theme]
st.markdown(f"""
<style>
.stApp {{background-color:{C['fond']}; color:{C['texte']};}}
[data-testid="stSidebar"] {{background-color:{C['panneau']};}}
[data-testid="stSidebar"] * {{color:{C['texte']};}}
h1,h2,h3,h4,p,label {{color:{C['texte']};}}
[data-testid="stMetric"] {{background:{C['panneau']};border:1px solid {C['bordure']};padding:15px;border-radius:12px;}}
div.stButton>button,div.stDownloadButton>button {{border-radius:8px;font-weight:600;}}
.hero {{background:linear-gradient(110deg,{C['primaire']},{C['accent']});padding:24px;border-radius:15px;margin-bottom:20px;}}
.hero h1,.hero p {{color:white !important;}}
</style>
""", unsafe_allow_html=True)

def style_graphique(fig, titre):
    fig.update_layout(
        title=titre, template=C["graphique"],
        font=dict(family="Arial", size=14, color=C["texte"]),
        paper_bgcolor=C["fond"], plot_bgcolor=C["fond"],
        margin=dict(l=25, r=25, t=65, b=35), height=400,
    )
    return fig

def graphique_histogramme(df):
    fig = px.histogram(df, x="strength_mpa", nbins=max(5, min(20, len(df))),
                       labels={"strength_mpa": "Résistance (MPa)"})
    return style_graphique(fig, T("chart_distribution"))

def graphique_evolution(df):
    ordre = df.sort_values("test_date")
    fig = px.line(ordre, x="test_date", y="strength_mpa", markers=True,
                  color="sample_code",
                  labels={"test_date": T("test_date"), "strength_mpa": "MPa", "sample_code": T("code")})
    return style_graphique(fig, T("chart_evolution"))

def graphique_boite(df):
    fig = px.box(df, y="strength_mpa", points="all",
                 labels={"strength_mpa": "Résistance (MPa)"})
    return style_graphique(fig, T("chart_box"))

# =========================================================
# 6. BARRE LATÉRALE
# =========================================================
with st.sidebar:
    st.title("🏗️ BetonLab")
    langue = st.selectbox(T("language"), ["Français", "English"],
                          index=0 if st.session_state.lang == "Français" else 1)
    if langue != st.session_state.lang:
        st.session_state.lang = langue
        st.rerun()

    theme = st.radio(T("appearance"), ["Clair", "Sombre"],
                     index=0 if st.session_state.theme == "Clair" else 1)
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()

    st.divider()
    pages = [T(k) for k in ["dashboard", "add", "database", "statistics", "probability", "charts", "history"]]
    page = st.radio("Navigation", pages)
    st.divider()

    if MODE_STOCKAGE == "PostgreSQL":
        st.success(T("storage_cloud"))
    else:
        st.info(T("storage_local"))
        st.warning(T("persistent_warning"))

    if st.button(T("reload_data"), use_container_width=True):
        st.rerun()
    if st.button(T("saved_data"), use_container_width=True):
        st.session_state["afficher_donnees_stockees"] = not st.session_state.get("afficher_donnees_stockees", False)

st.markdown(f"""
<div class="hero">
  <h1>🏗️ BetonLab</h1>
  <p>{T("title")}</p>
  <p>{T("footer")}</p>
</div>
""", unsafe_allow_html=True)

df = charger_essais()
s = calculer_statistiques(df)

if st.session_state.get("last_page") != page:
    st.session_state.last_page = page
    enregistrer_action("Consultation", page)

# =========================================================
# 7. TABLEAU DE BORD
# =========================================================
if page == T("dashboard"):
    st.header(T("overview"))
    if df.empty:
        st.info(T("no_data"))
        st.write(T("demo_intro"))
        if st.button(T("demo"), use_container_width=True):
            try:
                if ajouter_donnees_demo():
                    st.success(T("demo_added"))
                else:
                    st.info(T("demo_exists"))
                st.rerun()
            except Exception as exc:
                st.error(f"{T('error')}: {exc}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T("tests"), s["nombre"])
        c2.metric(T("mean"), f'{s["moyenne"]:.2f} MPa')
        c3.metric(T("maximum"), f'{s["maximum"]:.2f} MPa')
        c4.metric(T("minimum"), f'{s["minimum"]:.2f} MPa')
        gauche, droite = st.columns(2)
        with gauche:
            st.plotly_chart(graphique_histogramme(df), use_container_width=True)
        with droite:
            st.plotly_chart(graphique_evolution(df), use_container_width=True)
        st.subheader("Derniers essais" if st.session_state.lang == "Français" else "Latest tests")
        affichage = df.head(10).copy()
        affichage["test_date"] = affichage["test_date"].dt.strftime("%d/%m/%Y")
        st.dataframe(affichage, use_container_width=True, hide_index=True)

# =========================================================
# 8. ENREGISTREMENT
# =========================================================
elif page == T("add"):
    st.header(T("add"))
    projet = st.text_input(T("project_name"), value="Projet général" if st.session_state.lang == "Français" else "General project")

    with st.form("formulaire_essai", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            code = st.text_input(T("code"))
            jour = st.date_input(T("test_date"), value=date.today())
            age = st.number_input(T("age"), min_value=1, max_value=36500, value=28)
            forme_affichee = st.selectbox(T("shape"), [T("prismatic"), T("cylindrical")])
        with col2:
            forme_db = "Cylindrique" if forme_affichee == T("cylindrical") else "Prismatique"
            if forme_db == "Cylindrique":
                longueur = st.number_input(T("diameter"), min_value=0.1, value=150.0)
                largeur = longueur
            else:
                longueur = st.number_input(T("length"), min_value=0.1, value=150.0)
                largeur = st.number_input(T("width"), min_value=0.1, value=150.0)
            hauteur = st.number_input(T("height"), min_value=0.1, value=300.0)
            charge = st.number_input(T("load"), min_value=0.1, value=650.0)
        envoyer = st.form_submit_button(T("save"), use_container_width=True)

    if envoyer:
        if not code.strip():
            st.error(T("empty_code"))
        else:
            try:
                resistance = calculer_resistance(charge, longueur, largeur, forme_db)
                donnees = {
                    "sample_code": code.strip(), "test_date": jour.isoformat(),
                    "age_days": int(age), "length_mm": float(longueur),
                    "width_mm": float(largeur), "height_mm": float(hauteur),
                    "load_kn": float(charge), "strength_mpa": resistance,
                    "shape": forme_db, "project": projet.strip() or "Projet général",
                    "specimen_reference": code.strip(),
                }
                reference = inserer_essai(donnees)
                enregistrer_action("Ajout d'un essai", f"{code} — {resistance:.2f} MPa — référence {reference}")
                st.success(T("success"))
                if reference != code.strip():
                    st.info(f"Référence enregistrée : {reference}")
                st.metric(T("strength"), f"{resistance:.2f} MPa")
                st.rerun()
            except Exception as exc:
                st.error(f"{T('error')}: {exc}")

# =========================================================
# 9. BASE, IMPORTATION, EXPORTATION, SUPPRESSION
# =========================================================
elif page == T("database"):
    st.header(T("database"))
    fichier = st.file_uploader(T("import"), type=["csv"])
    if fichier is not None and st.button(T("import_button")):
        try:
            import_df = pd.read_csv(fichier)
            obligatoires = ["sample_code", "test_date", "age_days", "length_mm", "width_mm", "height_mm", "load_kn"]
            manquantes = [c for c in obligatoires if c not in import_df.columns]
            if manquantes:
                st.error(f"{T('missing_columns')}: {', '.join(manquantes)}")
            else:
                compteur = 0
                for _, ligne in import_df.iterrows():
                    forme = str(ligne.get("shape", "Prismatique"))
                    forme = "Cylindrique" if forme.lower() in ("cylindrique", "cylindrical") else "Prismatique"
                    longueur = float(ligne["length_mm"])
                    largeur = float(ligne["width_mm"])
                    charge = float(ligne["load_kn"])
                    resistance = ligne.get("strength_mpa", np.nan)
                    if pd.isna(resistance):
                        resistance = calculer_resistance(charge, longueur, largeur, forme)
                    inserer_essai({
                        "sample_code": str(ligne["sample_code"]),
                        "test_date": str(ligne["test_date"])[:10],
                        "age_days": int(ligne["age_days"]),
                        "length_mm": longueur, "width_mm": largeur,
                        "height_mm": float(ligne["height_mm"]),
                        "load_kn": charge, "strength_mpa": float(resistance),
                        "shape": forme, "project": str(ligne.get("project", "Import CSV")),
                        "specimen_reference": str(ligne.get("specimen_reference", ligne["sample_code"])),
                    })
                    compteur += 1
                enregistrer_action("Importation CSV", f"{compteur} essais importés")
                st.success(f"{T('import_success')} {compteur}")
                st.rerun()
        except Exception as exc:
            st.error(f"{T('error')}: {exc}")

    df = charger_essais()
    if df.empty:
        st.info(T("no_data"))
    else:
        recherche = st.text_input(T("search"))
        vue = df.copy()
        if recherche.strip():
            masque = (
                vue["sample_code"].astype(str).str.contains(recherche, case=False, na=False)
                | vue["specimen_reference"].astype(str).str.contains(recherche, case=False, na=False)
                | vue["project"].astype(str).str.contains(recherche, case=False, na=False)
            )
            vue = vue[masque]
        st.dataframe(vue, use_container_width=True, hide_index=True)
        st.download_button(T("export"), data=vue.to_csv(index=False).encode("utf-8-sig"),
                           file_name="betonlab_essais.csv", mime="text/csv")
        st.subheader(T("delete"))
        identifiant = st.number_input(T("delete_id"), min_value=1, step=1)
        confirmation = st.checkbox(T("confirm"))
        if st.button(T("delete")):
            if not confirmation:
                st.warning(T("confirm"))
            else:
                try:
                    with moteur.begin() as con:
                        resultat = con.execute(text("DELETE FROM tests WHERE id = :id"), {"id": int(identifiant)})
                    if resultat.rowcount:
                        enregistrer_action("Suppression d'un essai", f"ID {identifiant}")
                        st.success(T("deleted"))
                        st.rerun()
                    else:
                        st.error(T("invalid_id"))
                except Exception as exc:
                    st.error(f"{T('error')}: {exc}")

# =========================================================
# 10. STATISTIQUES
# =========================================================
elif page == T("statistics"):
    st.header(T("statistics"))
    if not s:
        st.info(T("no_data"))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T("tests"), s["nombre"])
        c2.metric("Médiane" if st.session_state.lang == "Français" else "Median", f'{s["mediane"]:.2f} MPa')
        c3.metric("Étendue" if st.session_state.lang == "Français" else "Range", f'{s["etendue"]:.2f} MPa')
        c4.metric("Écart-type" if st.session_state.lang == "Français" else "Standard deviation",
                  "—" if s["ecart_type"] is None else f'{s["ecart_type"]:.2f} MPa')
        c1, c2 = st.columns(2)
        c1.metric("Coefficient de variation" if st.session_state.lang == "Français" else "Coefficient of variation",
                  "—" if s["cv"] is None else f'{s["cv"]:.2f}%')
        c2.metric(T("maximum"), f'{s["maximum"]:.2f} MPa')
        st.plotly_chart(graphique_boite(df), use_container_width=True)
        valeurs = pd.to_numeric(df["strength_mpa"], errors="coerce").dropna().to_numpy()
        if len(valeurs) >= 3 and np.std(valeurs) > 0:
            resultat = stats.shapiro(valeurs[:5000])
            st.subheader(T("normality"))
            st.write(f"Statistic: {resultat.statistic:.4f}" if st.session_state.lang == "English" else f"Statistique : {resultat.statistic:.4f}")
            st.write(f"p-value: {resultat.pvalue:.4f}")
            if resultat.pvalue < 0.05:
                st.warning(T("normality_warning"))
            else:
                st.info(T("normality_ok"))

# =========================================================
# 11. PROBABILITÉS
# =========================================================
elif page == T("probability"):
    st.header(T("probability"))
    valeurs = pd.to_numeric(df["strength_mpa"], errors="coerce").dropna().to_numpy()
    if len(valeurs) < 2:
        st.info(T("need_two"))
    else:
        moyenne = float(np.mean(valeurs))
        ecart_type = float(np.std(valeurs, ddof=1))
        st.metric(T("mean"), f"{moyenne:.2f} MPa")
        if ecart_type == 0:
            st.warning(T("std_zero"))
        else:
            cible = st.number_input(T("target"), min_value=0.0, value=float(round(moyenne, 1)), step=0.5)
            p_inf = float(stats.norm.cdf(cible, loc=moyenne, scale=ecart_type))
            p_sup = 1.0 - p_inf
            c1, c2 = st.columns(2)
            c1.metric(T("below"), f"{100*p_inf:.2f}%")
            c2.metric(T("above"), f"{100*p_sup:.2f}%")
            xx = np.linspace(max(0, moyenne - 4*ecart_type), moyenne + 4*ecart_type, 400)
            yy = stats.norm.pdf(xx, loc=moyenne, scale=ecart_type)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="Normal distribution", line=dict(width=3)))
            fig.add_vline(x=cible, line_dash="dash", annotation_text="Target")
            st.plotly_chart(style_graphique(fig, "Estimated normal distribution"), use_container_width=True)
            st.caption(T("probability_note"))

# =========================================================
# 12. DIAGRAMMES
# =========================================================
elif page == T("charts"):
    st.header(T("charts"))
    if df.empty:
        st.info(T("no_data"))
    else:
        choix = st.selectbox(T("chart_type"), [T("histogram"), T("evolution"), T("boxplot"), T("age_chart")])
        if choix == T("histogram"):
            fig = graphique_histogramme(df)
        elif choix == T("evolution"):
            fig = graphique_evolution(df)
        elif choix == T("boxplot"):
            fig = graphique_boite(df)
        else:
            fig = px.scatter(df, x="age_days", y="strength_mpa", color="sample_code",
                             labels={"age_days": T("age"), "strength_mpa": "MPa", "sample_code": T("code")})
            fig = style_graphique(fig, T("age_chart"))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(T("export"), data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="betonlab_donnees_graphiques.csv", mime="text/csv")

# =========================================================
# 13. HISTORIQUE
# =========================================================
elif page == T("history"):
    st.header(T("history"))
    historique = charger_historique()
    st.metric(T("actions"), len(historique))
    if historique.empty:
        st.info(T("history_empty"))
    else:
        st.dataframe(historique, use_container_width=True, hide_index=True)
        st.download_button(T("export"), data=historique.to_csv(index=False).encode("utf-8-sig"),
                           file_name="betonlab_historique.csv", mime="text/csv")

# =========================================================
# 14. CONSULTATION DES DONNÉES DÉJÀ STOCKÉES
# =========================================================
if st.session_state.get("afficher_donnees_stockees", False):
    st.divider()
    st.subheader(T("saved_data_title"))
    st.caption(T("saved_data_help"))
    donnees_stockees = charger_essais()
    if donnees_stockees.empty:
        st.info(T("no_saved_data"))
    else:
        st.dataframe(donnees_stockees, use_container_width=True, hide_index=True)
        st.download_button(
            T("backup_csv"),
            data=donnees_stockees.to_csv(index=False).encode("utf-8-sig"),
            file_name="betonlab_sauvegarde.csv",
            mime="text/csv",
            key="download_backup_saved_data",
        )

# =========================================================
# 15. PIED DE PAGE
# =========================================================
st.divider()
st.caption(f"BetonLab • {T('title')} • Storage: {MODE_STOCKAGE} • Theme: {st.session_state.theme}")
