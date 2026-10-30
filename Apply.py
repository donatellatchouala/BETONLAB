"""BetonLab : laboratoire de résistance à la compression du béton.

- Base de données PostgreSQL (Neon) définie par DATABASE_URL dans les Secrets.
- Mode démonstration : une table séparée, remplie d'essais réalistes, pour
  découvrir l'application sans toucher aux vrais essais.
- Interface en français ou en anglais, thème clair ou sombre.
"""

import html
import io
import math
import os
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import make_url


st.set_page_config(
    page_title="BetonLab",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="auto",
)


# ============================================================
# 1. TRADUCTIONS (français / anglais)
# ============================================================

TR = {
    "fr": {
        "tagline": "Laboratoire de résistance du béton",
        "nav": "Navigation",
        "p_home": "Tableau de bord", "p_new": "Nouvel essai", "p_tests": "Essais",
        "p_stats": "Statistiques", "p_proba": "Probabilités", "p_charts": "Graphiques",
        "p_guide": "Guide", "p_settings": "Paramètres",
        "language": "Langue", "theme": "Thème", "light": "Clair", "dark": "Sombre",
        "demo_mode": "Données de démonstration",
        "demo_help": "Affiche une base d'essais fictifs mais réalistes pour découvrir l'application. "
                     "Vos vrais essais ne sont ni affichés ni modifiés.",
        "demo_chip": "Démonstration",
        "real_chip": "Essais réels",
        "footer": "Vérifiez toujours les unités et les normes applicables (EN 206, EN 12390).",
        # accueil
        "hero_title": "Analysez vos essais de compression en quelques clics",
        "hero_text": "BetonLab enregistre vos éprouvettes, calcule automatiquement leur résistance "
                     "et vous donne statistiques, graphiques et probabilités pour juger un béton.",
        "hero_demo": "Explorer avec des données de démonstration",
        "hero_new": "Enregistrer un essai",
        "how_title": "Comment ça marche",
        "step1_t": "Saisissez l'essai", "step1_d": "Code, date, âge, dimensions de l'éprouvette et charge de rupture.",
        "step2_t": "Résistance calculée", "step2_d": "f = F / A : la résistance en MPa est calculée et enregistrée.",
        "step3_t": "Analysez", "step3_d": "Moyenne, écart-type, fractile 5 %, évolution avec l'âge, comparaison des chantiers.",
        "step4_t": "Exportez", "step4_d": "Téléchargez vos essais en CSV pour Excel ou un rapport.",
        "kpi_tests": "Essais", "kpi_mean": "Résistance moyenne", "kpi_min": "Minimum",
        "kpi_max": "Maximum", "kpi_projects": "Chantiers",
        "kpi_tests_sub": "éprouvettes testées", "kpi_mean_sub": "toutes éprouvettes",
        "recent": "Derniers essais", "trend": "Résistances dans le temps",
        "empty_title": "Aucun essai pour le moment",
        "empty_text": "Enregistrez votre premier essai, ou activez les données de démonstration pour voir l'application en action.",
        # formulaire
        "new_title": "Nouvel essai", "new_sub": "La résistance est calculée automatiquement : f = F × 1000 / A (MPa).",
        "specimen": "Type d'éprouvette", "Cube": "Cube", "Cylindre": "Cylindre",
        "code": "Code de l'éprouvette", "project": "Chantier / projet", "project_default": "Projet général",
        "test_date": "Date de l'essai", "age": "Âge du béton (jours)", "diameter": "Diamètre (mm)",
        "length": "Longueur de la face chargée (mm)", "width": "Largeur de la face chargée (mm)",
        "load": "Charge de rupture (kN)", "notes": "Observations", "save": "Calculer et enregistrer",
        "saved": "Essai enregistré.", "strength": "Résistance", "area": "Surface chargée",
        "preview": "Aperçu du calcul",
        # base
        "tests_title": "Essais", "tests_sub": "Recherchez, filtrez par période, exportez ou supprimez.",
        "search": "Rechercher (code, chantier, observation)", "period": "Période",
        "shown": "{n} essai(s) affiché(s) sur {total}", "export": "Exporter en CSV",
        "delete_sel": "Supprimer des essais", "delete_pick": "Essais à supprimer",
        "delete_btn": "Supprimer la sélection", "deleted": "{n} essai(s) supprimé(s).",
        # colonnes
        "c_code": "Code", "c_project": "Chantier", "c_date": "Date", "c_age": "Âge (j)",
        "c_type": "Type", "c_load": "Charge (kN)", "c_area": "Surface (mm²)",
        "c_strength": "Résistance (MPa)", "c_notes": "Observations",
        # filtres / stats
        "filters": "Filtres", "projects": "Chantiers", "all": "Tous", "ages": "Âges (jours)",
        "mixed": "Cubes et cylindres sont mélangés : un cube donne environ 20 % de plus. "
                 "Choisissez un seul type pour comparer.",
        "no_match": "Aucun essai ne correspond aux filtres.",
        "stats_title": "Statistiques", "stats_sub": "Indicateurs calculés sur les essais filtrés.",
        "s_n": "Nombre", "s_mean": "Moyenne", "s_median": "Médiane", "s_sd": "Écart-type",
        "s_cv": "Coefficient de variation", "s_min": "Minimum", "s_max": "Maximum",
        "by_age": "Par âge", "by_project": "Par chantier", "help_stats":
        "Écart-type : dispersion des résultats. CV = écart-type / moyenne : plus il est faible, "
        "plus la fabrication est régulière.",
        # proba
        "proba_title": "Probabilités", "proba_sub": "Modèle de loi normale ajusté sur les essais filtrés.",
        "fk": "Fractile 5 %", "fk_help": "m − 1,645·s : valeur dépassée par 95 % des éprouvettes (loi normale).",
        "threshold": "Résistance seuil (MPa)", "p_above": "Probabilité de dépasser le seuil",
        "p_below": "Probabilité d'être sous le seuil", "need2": "Il faut au moins deux essais.",
        "sd0": "Tous les résultats sont identiques : la loi normale n'est pas exploitable.",
        "few": "Seulement {n} essais : ces probabilités sont indicatives.",
        "proba_note": "Estimations sous hypothèse de loi normale ; elles ne remplacent pas "
                      "le contrôle de conformité de la norme EN 206.",
        "density": "Densité", "count": "Nombre d'éprouvettes",
        # graphiques
        "charts_title": "Graphiques", "charts_sub": "Visualisez la répartition et l'évolution des résistances.",
        "g_hist": "Répartition des résistances", "g_age": "Montée en résistance avec l'âge",
        "g_project": "Comparaison des chantiers", "g_last": "30 derniers essais",
        "mean_line": "Moyenne",
        # guide
        "guide_title": "Guide d'utilisation",
        "guide_intro": "BetonLab sert à suivre la résistance à la compression du béton d'un chantier ou d'un "
                       "laboratoire : on écrase des éprouvettes (cubes ou cylindres) à différents âges, on note la "
                       "charge de rupture, et l'application fait tous les calculs.",
        "g1_t": "1. Enregistrer un essai",
        "g1_d": "Menu « Nouvel essai ». Choisissez cube ou cylindre, puis saisissez le code (unique), le chantier, "
                "la date, l'âge du béton, les dimensions en mm et la charge de rupture en kN. "
                "La résistance s'affiche avant même l'enregistrement.",
        "g2_t": "2. Retrouver et exporter",
        "g2_d": "Menu « Essais » : recherche par code ou chantier, filtre par période, export CSV (s'ouvre dans Excel).",
        "g3_t": "3. Lire les statistiques",
        "g3_d": "Moyenne : résistance typique. Écart-type : dispersion. CV : régularité de la fabrication "
                "(plus il est petit, mieux c'est). Filtrez par chantier, type et âge pour comparer ce qui est comparable.",
        "g4_t": "4. Probabilités",
        "g4_d": "Le fractile 5 % est la résistance dépassée par 95 % des éprouvettes. Entrez une résistance seuil "
                "(ex. la classe visée) pour connaître la probabilité d'être en dessous.",
        "g5_t": "5. Données de démonstration",
        "g5_d": "Activez-les dans le menu pour essayer l'application sur une base d'essais réalistes. "
                "Elles sont rangées à part : vos vrais essais ne sont jamais modifiés.",
        "g6_t": "Bon à savoir",
        "g6_d": "Un cube de 150 mm donne environ 20 % de plus qu'un cylindre de 160 × 320 mm : ne les mélangez pas "
                "dans une même analyse. La résistance à 28 jours est la référence des classes de béton (ex. C25/30).",
        # paramètres
        "settings_title": "Paramètres", "settings_sub": "Base de données, import, données de démonstration.",
        "db": "Base de données", "db_ok": "Connectée : les essais sont conservés de façon permanente.",
        "db_temp": "Base temporaire : ajoutez psycopg2-binary dans requirements.txt et DATABASE_URL dans les "
                   "Secrets, puis redémarrez l'application, pour conserver les essais.",
        "demo_box": "Données de démonstration",
        "demo_reload": "Recharger les données de démonstration",
        "demo_reloaded": "{n} essais de démonstration chargés.",
        "demo_only": "Activez les données de démonstration dans le menu pour les recharger.",
        "import": "Importer des essais (CSV)",
        "import_help": "Même format que l'export. Colonnes obligatoires : sample_code, test_date, age_days, "
                       "specimen_type, load_kn, et diameter_mm (cylindre) ou length_mm et width_mm (cube). "
                       "La résistance est recalculée.",
        "file": "Fichier CSV", "valid_rows": "{ok} ligne(s) valide(s), {bad} problème(s).",
        "see_problems": "Voir les problèmes", "import_btn": "Importer {n} essai(s)",
        "imported": "{n} essai(s) importé(s).",
        "danger": "Zone sensible", "delete_all": "Supprimer tous les essais de cette base",
        "delete_all_confirm": "Je confirme vouloir supprimer TOUS les essais affichés dans cette base.",
        "type_delete": "Écrivez SUPPRIMER pour confirmer",
        "deleted_all": "Tous les essais ont été supprimés.",
        "confirm_needed": "Cochez la confirmation et écrivez SUPPRIMER.",
        "conn_error": "Connexion à la base de données impossible. Vérifiez DATABASE_URL dans les Secrets.",
        "read_error": "Impossible de lire les essais.",
        "write_error": "Opération impossible.",
        # erreurs de saisie
        "e_code": "Le code de l'éprouvette est obligatoire.",
        "e_code_long": "Le code dépasse 100 caractères.",
        "e_project_long": "Le nom du chantier dépasse 200 caractères.",
        "e_date": "La date de l'essai est manquante.",
        "e_date_bad": "Date illisible : {v}",
        "e_future": "La date de l'essai est dans le futur.",
        "e_age": "L'âge doit être compris entre 0 et 3650 jours.",
        "e_age_missing": "L'âge du béton est manquant.",
        "e_load_missing": "La charge est manquante.",
        "e_load": "La charge doit être positive.",
        "e_diam": "Le diamètre doit être positif.",
        "e_dims": "La longueur et la largeur doivent être positives.",
        "e_type": "Type d'éprouvette inconnu : {v}",
        "e_absurd": "Résistance de {v} MPa : invraisemblable. Vérifiez la charge (kN) et les dimensions (mm).",
        "e_dup": "Le code {v} existe déjà. Choisissez un code unique.",
        "e_dup_line": "code {v} déjà présent, ligne ignorée.",
        "e_cols": "Colonnes manquantes : {v}.",
        "e_file": "Fichier illisible : enregistrez-le en CSV (UTF-8).",
        "line": "Ligne {n}",
        "dl_name": "betonlab_essais",
    },
    "en": {
        "tagline": "Concrete strength laboratory",
        "nav": "Navigation",
        "p_home": "Dashboard", "p_new": "New test", "p_tests": "Tests",
        "p_stats": "Statistics", "p_proba": "Probability", "p_charts": "Charts",
        "p_guide": "Guide", "p_settings": "Settings",
        "language": "Language", "theme": "Theme", "light": "Light", "dark": "Dark",
        "demo_mode": "Demo data",
        "demo_help": "Shows a realistic sample database to explore the app. "
                     "Your real tests are neither shown nor modified.",
        "demo_chip": "Demo",
        "real_chip": "Real tests",
        "footer": "Always check units and applicable standards (EN 206, EN 12390).",
        "hero_title": "Analyse your compression tests in a few clicks",
        "hero_text": "BetonLab records your specimens, computes their strength automatically and gives you "
                     "statistics, charts and probabilities to assess a concrete.",
        "hero_demo": "Explore with demo data",
        "hero_new": "Record a test",
        "how_title": "How it works",
        "step1_t": "Enter the test", "step1_d": "Code, date, age, specimen dimensions and failure load.",
        "step2_t": "Strength computed", "step2_d": "f = F / A: the strength in MPa is computed and saved.",
        "step3_t": "Analyse", "step3_d": "Mean, standard deviation, 5% fractile, strength gain with age, site comparison.",
        "step4_t": "Export", "step4_d": "Download your tests as CSV for Excel or a report.",
        "kpi_tests": "Tests", "kpi_mean": "Mean strength", "kpi_min": "Minimum",
        "kpi_max": "Maximum", "kpi_projects": "Sites",
        "kpi_tests_sub": "specimens tested", "kpi_mean_sub": "all specimens",
        "recent": "Latest tests", "trend": "Strength over time",
        "empty_title": "No tests yet",
        "empty_text": "Record your first test, or turn on the demo data to see the app in action.",
        "new_title": "New test", "new_sub": "Strength is computed automatically: f = F × 1000 / A (MPa).",
        "specimen": "Specimen type", "Cube": "Cube", "Cylindre": "Cylinder",
        "code": "Specimen code", "project": "Site / project", "project_default": "General project",
        "test_date": "Test date", "age": "Concrete age (days)", "diameter": "Diameter (mm)",
        "length": "Loaded face length (mm)", "width": "Loaded face width (mm)",
        "load": "Failure load (kN)", "notes": "Notes", "save": "Compute and save",
        "saved": "Test saved.", "strength": "Strength", "area": "Loaded area",
        "preview": "Calculation preview",
        "tests_title": "Tests", "tests_sub": "Search, filter by period, export or delete.",
        "search": "Search (code, site, notes)", "period": "Period",
        "shown": "{n} test(s) shown out of {total}", "export": "Export as CSV",
        "delete_sel": "Delete tests", "delete_pick": "Tests to delete",
        "delete_btn": "Delete selection", "deleted": "{n} test(s) deleted.",
        "c_code": "Code", "c_project": "Site", "c_date": "Date", "c_age": "Age (d)",
        "c_type": "Type", "c_load": "Load (kN)", "c_area": "Area (mm²)",
        "c_strength": "Strength (MPa)", "c_notes": "Notes",
        "filters": "Filters", "projects": "Sites", "all": "All", "ages": "Ages (days)",
        "mixed": "Cubes and cylinders are mixed: a cube gives about 20% more. "
                 "Choose a single type to compare.",
        "no_match": "No test matches the filters.",
        "stats_title": "Statistics", "stats_sub": "Indicators computed on the filtered tests.",
        "s_n": "Count", "s_mean": "Mean", "s_median": "Median", "s_sd": "Standard deviation",
        "s_cv": "Coefficient of variation", "s_min": "Minimum", "s_max": "Maximum",
        "by_age": "By age", "by_project": "By site", "help_stats":
        "Standard deviation: spread of the results. CV = standard deviation / mean: the lower it is, "
        "the more consistent the production.",
        "proba_title": "Probability", "proba_sub": "Normal distribution fitted to the filtered tests.",
        "fk": "5% fractile", "fk_help": "m − 1.645·s: value exceeded by 95% of specimens (normal distribution).",
        "threshold": "Threshold strength (MPa)", "p_above": "Probability of exceeding the threshold",
        "p_below": "Probability of falling below", "need2": "At least two tests are needed.",
        "sd0": "All results are identical: the normal distribution cannot be used.",
        "few": "Only {n} tests: these probabilities are indicative.",
        "proba_note": "Estimates assuming a normal distribution; they do not replace "
                      "the EN 206 conformity assessment.",
        "density": "Density", "count": "Number of specimens",
        "charts_title": "Charts", "charts_sub": "See how strengths are distributed and how they develop.",
        "g_hist": "Strength distribution", "g_age": "Strength gain with age",
        "g_project": "Site comparison", "g_last": "Last 30 tests",
        "mean_line": "Mean",
        "guide_title": "User guide",
        "guide_intro": "BetonLab tracks the compressive strength of concrete on a site or in a laboratory: "
                       "specimens (cubes or cylinders) are crushed at various ages, the failure load is recorded, "
                       "and the app does all the calculations.",
        "g1_t": "1. Record a test",
        "g1_d": "“New test” menu. Choose cube or cylinder, then enter the (unique) code, the site, the date, "
                "the concrete age, the dimensions in mm and the failure load in kN. "
                "The strength is shown before you save.",
        "g2_t": "2. Find and export",
        "g2_d": "“Tests” menu: search by code or site, filter by period, CSV export (opens in Excel).",
        "g3_t": "3. Read the statistics",
        "g3_d": "Mean: typical strength. Standard deviation: spread. CV: production consistency "
                "(the smaller the better). Filter by site, type and age to compare like with like.",
        "g4_t": "4. Probability",
        "g4_d": "The 5% fractile is the strength exceeded by 95% of specimens. Enter a threshold strength "
                "(e.g. the target class) to get the probability of falling below it.",
        "g5_t": "5. Demo data",
        "g5_d": "Turn them on in the menu to try the app on a realistic test database. "
                "They are stored separately: your real tests are never modified.",
        "g6_t": "Good to know",
        "g6_d": "A 150 mm cube gives about 20% more than a 160 × 320 mm cylinder: do not mix them in one analysis. "
                "The 28-day strength is the reference for concrete classes (e.g. C25/30).",
        "settings_title": "Settings", "settings_sub": "Database, import, demo data.",
        "db": "Database", "db_ok": "Connected: tests are stored permanently.",
        "db_temp": "Temporary database: add psycopg2-binary to requirements.txt and DATABASE_URL to the "
                   "Secrets, then reboot the app, to keep your tests.",
        "demo_box": "Demo data",
        "demo_reload": "Reload the demo data",
        "demo_reloaded": "{n} demo tests loaded.",
        "demo_only": "Turn on the demo data in the menu to reload it.",
        "import": "Import tests (CSV)",
        "import_help": "Same format as the export. Required columns: sample_code, test_date, age_days, "
                       "specimen_type, load_kn, and diameter_mm (cylinder) or length_mm and width_mm (cube). "
                       "Strength is recomputed.",
        "file": "CSV file", "valid_rows": "{ok} valid row(s), {bad} problem(s).",
        "see_problems": "See the problems", "import_btn": "Import {n} test(s)",
        "imported": "{n} test(s) imported.",
        "danger": "Danger zone", "delete_all": "Delete all tests in this database",
        "delete_all_confirm": "I confirm I want to delete ALL tests in this database.",
        "type_delete": "Type DELETE to confirm",
        "deleted_all": "All tests have been deleted.",
        "confirm_needed": "Tick the confirmation and type DELETE.",
        "conn_error": "Cannot connect to the database. Check DATABASE_URL in the Secrets.",
        "read_error": "Cannot read the tests.",
        "write_error": "Operation failed.",
        "e_code": "The specimen code is required.",
        "e_code_long": "The code is longer than 100 characters.",
        "e_project_long": "The site name is longer than 200 characters.",
        "e_date": "The test date is missing.",
        "e_date_bad": "Unreadable date: {v}",
        "e_future": "The test date is in the future.",
        "e_age": "Age must be between 0 and 3650 days.",
        "e_age_missing": "The concrete age is missing.",
        "e_load_missing": "The load is missing.",
        "e_load": "The load must be positive.",
        "e_diam": "The diameter must be positive.",
        "e_dims": "Length and width must be positive.",
        "e_type": "Unknown specimen type: {v}",
        "e_absurd": "Strength of {v} MPa: implausible. Check the load (kN) and the dimensions (mm).",
        "e_dup": "Code {v} already exists. Choose a unique code.",
        "e_dup_line": "code {v} already exists, row skipped.",
        "e_cols": "Missing columns: {v}.",
        "e_file": "Unreadable file: save it as CSV (UTF-8).",
        "line": "Row {n}",
        "dl_name": "betonlab_tests",
    },
}


def lang():
    return st.session_state.get("lang", "fr")


def t(key, **kw):
    text_value = TR[lang()].get(key, TR["fr"].get(key, key))
    return text_value.format(**kw) if kw else text_value


def num(x, digits=1):
    """Nombre formaté selon la langue (virgule en français)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:,.{digits}f}".replace(",", " ")
    return s.replace(".", ",") if lang() == "fr" else s.replace(" ", ",")


def fdate(d):
    if d is None or pd.isna(d):
        return "—"
    return d.strftime("%d/%m/%Y") if lang() == "fr" else d.isoformat()


# ============================================================
# 2. PRÉFÉRENCES : LANGUE ET THÈME (conservées dans l'adresse)
# ============================================================

for _key, _allowed, _default in (("lang", ("fr", "en"), "fr"), ("theme", ("light", "dark"), "light")):
    if _key not in st.session_state:
        _value = st.query_params.get(_key, _default)
        st.session_state[_key] = _value if _value in _allowed else _default

if st.session_state.pop("_go_demo", False):
    st.session_state["demo"] = True
    st.session_state["page"] = "p_home"
if "_go_page" in st.session_state:
    st.session_state["page"] = st.session_state.pop("_go_page")

PALETTES = {
    "light": {
        "bg": "#F3F6FB", "surface": "#FFFFFF", "surface2": "#F7F9FC", "text": "#13233A",
        "muted": "#5B6B82", "border": "#E2E8F0", "primary": "#1E5AA8", "primary2": "#0F3D73",
        "accent": "#F59E0B", "good": "#15803D", "bad": "#C2410C", "input": "#FFFFFF",
        "hero1": "#0F3D73", "hero2": "#1E6FB8", "grid": "#E6ECF3", "shadow": "0 1px 2px rgba(16,35,58,.06), 0 8px 24px rgba(16,35,58,.06)",
        "series": ["#1E5AA8", "#F59E0B", "#14B8A6", "#E11D48", "#7C3AED", "#64748B"],
    },
    "dark": {
        "bg": "#0B1220", "surface": "#121C2F", "surface2": "#17233A", "text": "#E6EDF7",
        "muted": "#9AABC3", "border": "#23314D", "primary": "#5B9BF0", "primary2": "#3B7DD8",
        "accent": "#FBBF24", "good": "#4ADE80", "bad": "#FB923C", "input": "#0F1829",
        "hero1": "#10264A", "hero2": "#1A4A86", "grid": "#22304A", "shadow": "0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25)",
        "series": ["#5B9BF0", "#FBBF24", "#2DD4BF", "#FB7185", "#A78BFA", "#94A3B8"],
    },
}
P = PALETTES[st.session_state["theme"]]


def inject_css():
    p = P
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {{
  --bg:{p['bg']}; --surface:{p['surface']}; --surface2:{p['surface2']}; --text:{p['text']};
  --muted:{p['muted']}; --border:{p['border']}; --primary:{p['primary']}; --primary2:{p['primary2']};
  --accent:{p['accent']}; --good:{p['good']}; --bad:{p['bad']}; --input:{p['input']}; --shadow:{p['shadow']};
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp td, .stApp th, .stApp span:not([data-testid="stIconMaterial"]) {{
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: var(--bg); color: var(--text); }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] * {{ color: var(--muted) !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: var(--text) !important; letter-spacing: -.01em; }}
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stCaptionContainer"],
.stApp label p, .stApp [role="radiogroup"] p, .stApp [data-testid="stCheckbox"] p {{ color: var(--text); }}
.stApp [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}

/* menu latéral */
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 8px 12px; border-radius: 10px; width: 100%; transition: background .15s;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--surface2); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: color-mix(in srgb, var(--primary) 14%, transparent);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: var(--primary) !important; font-weight: 700; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none; }}

/* champs */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="textarea"],
.stApp [data-baseweb="select"] > div, .stApp [data-testid="stNumberInputContainer"],
.stApp [data-testid="stTextInputRootElement"], .stApp [data-testid="stTextArea"] div:has(> textarea),
.stApp [data-testid="stDateInputField"] {{
  background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px;
}}
.stApp input, .stApp textarea {{ color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; background: transparent !important; }}
.stApp [data-baseweb="select"] * {{ color: var(--text); }}
.stApp [data-testid="stSelectbox"] [role="group"], .stApp [data-testid="stMultiSelect"] [role="group"] {{
  background: var(--input) !important; border-color: var(--border) !important; border-radius: 10px; }}
.stApp [data-testid="stSelectbox"] [role="group"] *:not(svg):not(path), .stApp [data-testid="stMultiSelect"] input {{ color: var(--text); }}
[role="listbox"], [role="listbox"] [role="option"] {{ background: var(--surface) !important; color: var(--text) !important; }}
[role="listbox"] [role="option"]:hover, [role="listbox"] [aria-selected="true"] {{ background: var(--surface2) !important; }}
.stApp [data-testid="stNumberInputStepDown"], .stApp [data-testid="stNumberInputStepUp"] {{
  background: var(--surface2) !important; color: var(--text) !important;
}}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"], [data-baseweb="calendar"],
[data-baseweb="calendar"] div, [data-baseweb="calendar"] button {{ background: var(--surface) !important; color: var(--text) !important; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] {{ background: var(--surface2) !important; }}
.stApp [data-baseweb="tag"] {{ background: color-mix(in srgb, var(--primary) 18%, transparent) !important; }}
.stApp [data-baseweb="tag"] span {{ color: var(--text) !important; }}
.stApp [data-testid="stFileUploaderDropzone"] {{ background: var(--surface2); border: 1px dashed var(--border); }}
.stApp [data-testid="stFileUploaderDropzone"] * {{ color: var(--muted); }}
.stApp [data-testid="stFileUploaderDropzone"] button {{ background: var(--surface); border: 1px solid var(--border); }}
.stApp [data-testid="stFileUploaderDropzone"] button * {{ color: var(--text); }}
.stApp input::placeholder, .stApp textarea::placeholder {{ color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: .7; }}

/* boutons */
.stApp .stButton > button, .stApp .stFormSubmitButton > button, .stApp .stDownloadButton > button {{
  border-radius: 10px; font-weight: 600; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); transition: all .15s;
}}
.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {{ border-color: var(--primary); color: var(--primary); }}
.stApp .stButton > button[kind="primary"], .stApp .stFormSubmitButton > button, .stApp .stDownloadButton > button {{
  background: linear-gradient(135deg, var(--primary), var(--primary2)); color: #fff !important; border: none;
  box-shadow: 0 6px 16px color-mix(in srgb, var(--primary) 30%, transparent);
}}
.stApp .stButton > button[kind="primary"] p, .stApp .stFormSubmitButton > button p, .stApp .stDownloadButton > button p {{ color: #fff !important; }}
.stApp .stButton > button[kind="primary"]:hover, .stApp .stFormSubmitButton > button:hover {{ filter: brightness(1.08); color: #fff; }}
.stApp [data-testid="stButtonGroup"] button {{ background: var(--surface2); border-color: var(--border); }}
.stApp [data-testid="stButtonGroup"] button p {{ color: var(--text); }}
.stApp [data-testid="stButtonGroup"] button[kind*="Active"] {{ background: var(--primary); border-color: var(--primary); }}
.stApp [data-testid="stButtonGroup"] button[kind*="Active"] p {{ color: #fff !important; }}

/* conteneurs */
.stApp [data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }}
.stApp [data-testid="stExpander"] summary {{ background: var(--surface) !important; border-radius: 14px; }}
.stApp [data-testid="stExpander"] summary:hover {{ background: var(--surface2) !important; }}
.stApp [data-testid="stExpander"] summary * {{ color: var(--text); }}
.stApp [data-testid="stDateInputField"] [role="group"] {{ background: transparent !important; }}
.stApp [data-testid="stDateInputField"] * {{ color: var(--text); }}
.stApp [data-testid="stForm"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; box-shadow: var(--shadow); }}
.stApp [data-testid="stAlertContainer"] {{ border-radius: 12px; }}
.stApp [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--border) !important; }}
hr {{ border-color: var(--border) !important; }}
code {{ background: var(--surface2) !important; color: var(--text) !important; }}

/* composants BetonLab */
.bl-brand {{ display:flex; align-items:center; gap:10px; margin: 4px 0 2px; }}
.bl-logo {{ width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-size:20px;
  background: linear-gradient(135deg, var(--primary), var(--accent)); box-shadow: var(--shadow); }}
.bl-brand b {{ font-size: 1.25rem; color: var(--text); letter-spacing: -.02em; }}
.bl-brand small {{ display:block; color: var(--muted); font-size: .78rem; font-weight: 500; }}
.bl-hero {{ border-radius: 22px; padding: 30px 32px; color: #fff; position: relative; overflow: hidden;
  background: radial-gradient(circle at 85% 20%, rgba(255,255,255,.14), transparent 40%),
              linear-gradient(135deg, {p['hero1']}, {p['hero2']}); box-shadow: var(--shadow); margin-bottom: 18px; }}
.bl-hero h1 {{ color: #fff !important; font-size: 1.9rem; line-height: 1.2; margin: 6px 0 10px; }}
.bl-hero p {{ color: rgba(255,255,255,.88) !important; font-size: 1.02rem; max-width: 720px; margin: 0; }}
.bl-hero .bl-chip {{ background: rgba(255,255,255,.16); color: #fff; border-color: rgba(255,255,255,.25); }}
.bl-chip {{ display:inline-flex; align-items:center; gap:6px; padding: 4px 11px; border-radius: 999px; font-size: .78rem;
  font-weight: 700; letter-spacing: .02em; border: 1px solid var(--border); background: var(--surface2); color: var(--muted); }}
.bl-chip.demo {{ background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent);
  border-color: color-mix(in srgb, var(--accent) 40%, transparent); }}
.bl-head {{ display:flex; justify-content:space-between; align-items:flex-end; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }}
.bl-head h1 {{ font-size: 1.85rem; margin: 0 !important; padding: 0 !important; }}
.bl-head p {{ color: var(--muted) !important; margin: 4px 0 0; }}
.bl-kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; margin: 6px 0 18px; }}
.bl-kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 16px 18px; box-shadow: var(--shadow); }}
.bl-kpi .lbl {{ color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }}
.bl-kpi .val {{ font-size: 1.75rem; font-weight: 800; color: var(--text); margin-top: 4px; letter-spacing: -.02em; }}
.bl-kpi .val small {{ font-size: .95rem; font-weight: 600; color: var(--muted); margin-left: 4px; }}
.bl-kpi .sub {{ color: var(--muted); font-size: .8rem; margin-top: 2px; }}
.bl-kpi.accent {{ border-top: 3px solid var(--accent); }}
.bl-kpi.primary {{ border-top: 3px solid var(--primary); }}
.bl-steps {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 8px 0 22px; }}
.bl-step {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 16px; box-shadow: var(--shadow); }}
.bl-step .n {{ width: 30px; height: 30px; border-radius: 9px; display:grid; place-items:center; font-weight: 800;
  background: color-mix(in srgb, var(--primary) 14%, transparent); color: var(--primary); margin-bottom: 10px; }}
.bl-step b {{ color: var(--text); display:block; margin-bottom: 4px; }}
.bl-step span {{ color: var(--muted); font-size: .9rem; }}
.bl-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 18px 20px; box-shadow: var(--shadow); margin-bottom: 14px; }}
.bl-card h3 {{ font-size: 1.05rem; margin: 0 0 10px !important; padding: 0 !important; }}
.bl-card p {{ color: var(--muted) !important; margin: 0; }}
.bl-section {{ font-weight: 700; font-size: 1.05rem; color: var(--text); margin: 18px 0 8px; }}
.bl-tablewrap {{ border: 1px solid var(--border); border-radius: 14px; overflow: auto; max-height: 520px; background: var(--surface); box-shadow: var(--shadow); }}
table.bl-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
table.bl-table th {{ position: sticky; top: 0; background: var(--surface2); color: var(--muted); text-align: left;
  font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; padding: 10px 10px; border-bottom: 1px solid var(--border); }}
table.bl-table td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); color: var(--text); white-space: nowrap; }}
table.bl-table tr:last-child td {{ border-bottom: none; }}
table.bl-table tr:hover td {{ background: var(--surface2); }}
table.bl-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.bl-table td.strong {{ font-weight: 700; }}
.bl-badge {{ display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: .76rem; font-weight: 700; }}
.bl-badge.cube {{ background: color-mix(in srgb, var(--primary) 15%, transparent); color: var(--primary); }}
.bl-badge.cyl {{ background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--accent); }}
.bl-result {{ border-radius: 16px; padding: 18px 20px; background: linear-gradient(135deg, {p['hero1']}, {p['hero2']}); color: #fff; box-shadow: var(--shadow); }}
.bl-result .lbl {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .8; font-weight: 700; }}
.bl-result .val {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em; }}
.bl-result .sub {{ opacity: .85; font-size: .9rem; }}
.bl-foot {{ color: var(--muted); font-size: .76rem; line-height: 1.4; }}
.bl-guide {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
</style>
""", unsafe_allow_html=True)


inject_css()


# ============================================================
# 3. COMPOSANTS D'AFFICHAGE
# ============================================================

E = html.escape


def page_header(title, subtitle=""):
    chip = (f'<span class="bl-chip demo">● {E(t("demo_chip"))}</span>' if st.session_state.get("demo")
            else f'<span class="bl-chip">● {E(t("real_chip"))}</span>')
    st.markdown(f'<div class="bl-head"><div><h1>{E(title)}</h1>'
                f'{f"<p>{E(subtitle)}</p>" if subtitle else ""}</div>{chip}</div>', unsafe_allow_html=True)


def kpis(items):
    """items : liste de (libellé, valeur, unité, sous-titre, style)."""
    cards = "".join(
        f'<div class="bl-kpi {style}"><div class="lbl">{E(lbl)}</div>'
        f'<div class="val">{E(val)}{f"<small>{E(unit)}</small>" if unit else ""}</div>'
        f'{f"<div class=sub>{E(sub)}</div>" if sub else ""}</div>'
        for lbl, val, unit, sub, style in items)
    st.markdown(f'<div class="bl-kpis">{cards}</div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="bl-section">{E(title)}</div>', unsafe_allow_html=True)


def tests_table(df, limit=None):
    """Tableau des essais, aux couleurs du thème."""
    rows = df if limit is None else df.head(limit)
    head = "".join(f"<th>{E(t(k))}</th>" for k in
                   ("c_code", "c_project", "c_date", "c_age", "c_type", "c_load", "c_strength", "c_notes"))
    body = []
    for r in rows.itertuples():
        badge = "cube" if r.specimen_type == "Cube" else "cyl"
        body.append(
            f"<tr><td class='strong'>{E(str(r.sample_code))}</td><td>{E(str(r.project))}</td>"
            f"<td>{E(fdate(r.test_date))}</td><td class='num'>{int(r.age_days)}</td>"
            f"<td><span class='bl-badge {badge}'>{E(t(r.specimen_type))}</span></td>"
            f"<td class='num'>{E(num(r.load_kn, 1))}</td><td class='num strong'>{E(num(r.strength_mpa, 2))}</td>"
            f"<td>{E(str(r.notes or ''))}</td></tr>")
    st.markdown(f'<div class="bl-tablewrap"><table class="bl-table"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)


def simple_table(columns, rows):
    head = "".join(f"<th>{E(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(
        f"<td class='{'num' if isinstance(v, (int, float, np.floating)) or (isinstance(v, str) and v[:1].isdigit()) else ''}'>"
        f"{E(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    st.markdown(f'<div class="bl-tablewrap"><table class="bl-table"><thead><tr>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)


FONT = "Plus Jakarta Sans, system-ui, sans-serif"


def show_chart(chart, height=320):
    """Graphique Altair aux couleurs du thème (clair ou sombre)."""
    chart = (chart.properties(height=height, background="transparent")
             .configure_view(stroke=None)
             .configure_axis(labelColor=P["muted"], titleColor=P["muted"], gridColor=P["grid"],
                             domainColor=P["border"], tickColor=P["border"], labelFont=FONT,
                             titleFont=FONT, titleFontWeight=600)
             .configure_legend(labelColor=P["text"], titleColor=P["muted"], labelFont=FONT,
                               titleFont=FONT, orient="bottom")
             .configure_title(color=P["text"], font=FONT))
    st.altair_chart(chart, theme=None, width="stretch")


def flash(kind, message):
    st.session_state["_flash"] = (kind, message)


def show_flash():
    item = st.session_state.pop("_flash", None)
    if item:
        st.toast(item[1], icon="✅" if item[0] == "success" else "⚠️")


# ============================================================
# 4. BASE DE DONNÉES
# ============================================================

TABLES = {"real": "beton_tests", "demo": "beton_tests_demo"}
SPECIMEN_TYPES = ["Cube", "Cylindre"]
COLUMNS = ["id", "sample_code", "project", "test_date", "age_days", "specimen_type", "diameter_mm",
           "length_mm", "width_mm", "load_kn", "area_mm2", "strength_mpa", "notes", "created_at"]
FALLBACK_URL = "sqlite:///betonlab_data.db"


def get_setting(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # pas de fichier de secrets
    return os.getenv(name, default)


def postgres_driver():
    for module in ("psycopg2", "psycopg"):
        try:
            __import__(module)
            return module
        except ImportError:
            continue
    return None


def database_url():
    """(adresse, permanente ?). PostgreSQL si possible, sinon base de secours sur le serveur."""
    url = str(get_setting("DATABASE_URL", "") or "").strip().strip('"').strip("'")
    driver = postgres_driver()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix) and driver:
            return f"postgresql+{driver}://" + url[len(prefix):], True
    return FALLBACK_URL, False


def safe_error(error, url):
    message = (str(error).strip().splitlines() or [""])[0]
    try:
        password = make_url(url).password
        if password:
            message = message.replace(password, "****")
    except Exception:
        pass
    return f"{type(error).__name__}: {message[:300]}"


@st.cache_resource(show_spinner=False)
def get_engine(url):
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800, pool_size=3, max_overflow=2,
                         connect_args={"connect_timeout": 10})


DDL = {
    "postgresql": """CREATE TABLE IF NOT EXISTS {table} (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, sample_code VARCHAR(100) NOT NULL,
        project VARCHAR(200) DEFAULT 'Projet général', test_date DATE NOT NULL, age_days INTEGER NOT NULL,
        specimen_type VARCHAR(30) NOT NULL, diameter_mm DOUBLE PRECISION, length_mm DOUBLE PRECISION,
        width_mm DOUBLE PRECISION, load_kn DOUBLE PRECISION NOT NULL, area_mm2 DOUBLE PRECISION NOT NULL,
        strength_mpa DOUBLE PRECISION NOT NULL, notes TEXT DEFAULT '', created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)""",
    "sqlite": """CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT, sample_code TEXT NOT NULL, project TEXT DEFAULT 'Projet général',
        test_date DATE NOT NULL, age_days INTEGER NOT NULL, specimen_type TEXT NOT NULL, diameter_mm REAL,
        length_mm REAL, width_mm REAL, load_kn REAL NOT NULL, area_mm2 REAL NOT NULL, strength_mpa REAL NOT NULL,
        notes TEXT DEFAULT '', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
}


@st.cache_resource(show_spinner=False)
def init_database(url):
    engine = get_engine(url)
    ddl = DDL["postgresql" if engine.dialect.name == "postgresql" else "sqlite"]
    with engine.begin() as connection:
        for table in TABLES.values():
            connection.execute(text(ddl.format(table=table)))
            connection.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_date ON {table} (test_date)"))
    return True


INSERT_SQL = """INSERT INTO {table} (sample_code, project, test_date, age_days, specimen_type, diameter_mm,
    length_mm, width_mm, load_kn, area_mm2, strength_mpa, notes) VALUES (:sample_code, :project, :test_date,
    :age_days, :specimen_type, :diameter_mm, :length_mm, :width_mm, :load_kn, :area_mm2, :strength_mpa, :notes)"""


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tests(url, table):
    with get_engine(url).connect() as connection:
        df = pd.read_sql(text(f"SELECT {', '.join(COLUMNS)} FROM {table} ORDER BY test_date DESC, id DESC"), connection)
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce").dt.date
    for column in ("age_days", "diameter_mm", "length_mm", "width_mm", "load_kn", "area_mm2", "strength_mpa"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["notes"] = df["notes"].fillna("")
    df["project"] = df["project"].fillna("").astype(str)
    return df


def save_tests(engine, table, records):
    if records:
        with engine.begin() as connection:
            connection.execute(text(INSERT_SQL.format(table=table)), records)
    fetch_tests.clear()
    return len(records)


def existing_codes(engine, table):
    with engine.connect() as connection:
        return {str(r[0]).strip().lower() for r in connection.execute(text(f"SELECT sample_code FROM {table}"))}


def delete_tests(engine, table, ids):
    statement = text(f"DELETE FROM {table} WHERE id IN :ids").bindparams(bindparam("ids", expanding=True))
    with engine.begin() as connection:
        result = connection.execute(statement, {"ids": [int(i) for i in ids]})
    fetch_tests.clear()
    return result.rowcount


def delete_all(engine, table):
    with engine.begin() as connection:
        connection.execute(text(f"DELETE FROM {table}"))
    fetch_tests.clear()


# ============================================================
# 5. CALCULS ET VÉRIFICATIONS
# ============================================================

def parse_date(value):
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw or raw.lower() in ("nan", "nat", "none"):
        raise ValueError(t("e_date"))
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(t("e_date_bad", v=raw)) from None
        return parsed.date()


def calculate_area(specimen_type, diameter=None, length=None, width=None):
    if specimen_type == "Cylindre":
        if diameter is None or not diameter > 0:
            raise ValueError(t("e_diam"))
        return math.pi * diameter ** 2 / 4
    if specimen_type == "Cube":
        if length is None or width is None or not (length > 0 and width > 0):
            raise ValueError(t("e_dims"))
        return length * width
    raise ValueError(t("e_type", v=specimen_type))


def build_record(sample_code, project, test_date, age_days, specimen_type, diameter, length, width, load_kn, notes):
    """Vérifie une saisie et renvoie la ligne prête à enregistrer (surface et résistance recalculées)."""
    sample_code = str(sample_code or "").strip()
    if not sample_code:
        raise ValueError(t("e_code"))
    if len(sample_code) > 100:
        raise ValueError(t("e_code_long"))
    project = str(project or "").strip() or t("project_default")
    if len(project) > 200:
        raise ValueError(t("e_project_long"))
    test_date = parse_date(test_date)
    if test_date > date.today():
        raise ValueError(t("e_future"))
    if age_days is None:
        raise ValueError(t("e_age_missing"))
    age_days = int(float(age_days))
    if not 0 <= age_days <= 3650:
        raise ValueError(t("e_age"))
    if specimen_type == "Cylindre":
        length = width = None
    else:
        diameter = None
    if load_kn is None:
        raise ValueError(t("e_load_missing"))
    load_kn = float(load_kn)
    if not load_kn > 0:
        raise ValueError(t("e_load"))
    area = calculate_area(specimen_type, diameter, length, width)
    strength = load_kn * 1000 / area  # kN × 1000 / mm² = N/mm² = MPa
    if strength > 200:
        raise ValueError(t("e_absurd", v=num(strength, 1)))
    return {
        "sample_code": sample_code, "project": project, "test_date": test_date.isoformat(), "age_days": age_days,
        "specimen_type": specimen_type,
        "diameter_mm": float(diameter) if diameter is not None else None,
        "length_mm": float(length) if length is not None else None,
        "width_mm": float(width) if width is not None else None,
        "load_kn": load_kn, "area_mm2": float(area), "strength_mpa": float(strength),
        "notes": str(notes or "").strip(),
    }


# ============================================================
# 6. DONNÉES DE DÉMONSTRATION
# ============================================================

DEMO_PROJECTS = [
    # (chantier, préfixe, classe : résistance caractéristique sur cylindre, éprouvette, ciment rapide ?)
    ("Pont sur le Nyong", "PNY", 30, "Cylindre", False),
    ("Immeuble R+5 Bastos", "BAS", 25, "Cube", False),
    ("École de Mvog-Ada — dalle", "EMA", 25, "Cylindre", True),
    ("Château d'eau de Soa", "SOA", 30, "Cube", False),
]
DEMO_NOTES = ["", "", "", "Rupture en double cône (normale)", "Éprouvette bien surfacée",
              "Légère ségrégation en surface", "", "Rupture conforme", ""]


def demo_records():
    """Environ 90 essais fictifs mais réalistes, toujours les mêmes (graine fixe)."""
    rng = np.random.default_rng(2026)
    gain = {3: 0.47, 7: 0.68, 14: 0.86, 28: 1.0, 90: 1.12}
    today = date.today()
    records = []
    for name, prefix, fck, specimen, fast in DEMO_PROJECTS:
        fcm28 = fck + 8.0  # résistance moyenne visée à 28 jours (EN 1992 : fcm = fck + 8)
        if specimen == "Cube":
            fcm28 *= 1.22  # un cube donne environ 20 % de plus qu'un cylindre
        for batch in range(1, 6):
            cast = today - timedelta(days=int(150 - batch * 22 + rng.integers(0, 6)))
            for age in (3, 7, 14, 28, 90) if fast else (7, 28, 90):
                test_day = cast + timedelta(days=age)
                if test_day > today:
                    continue
                factor = gain[age] * (1.05 if fast and age <= 7 else 1.0)
                for k in range(1, 3 if age != 28 else 4):
                    strength = max(6.0, rng.normal(fcm28 * factor, 2.6 if age >= 28 else 2.0))
                    if prefix == "BAS" and batch == 4 and age == 28 and k == 1:
                        strength -= 7.5  # un résultat faible, comme sur un vrai chantier
                    if specimen == "Cylindre":
                        diameter, length, width = 160.0, None, None
                        area = math.pi * diameter ** 2 / 4
                    else:
                        diameter, length, width = None, 150.0, 150.0
                        area = length * width
                    load = round(strength * area / 1000, 1)
                    records.append(build_record(
                        f"{prefix}-L{batch}-J{age}-{k}", name, test_day, age, specimen,
                        diameter, length, width, load, DEMO_NOTES[int(rng.integers(0, len(DEMO_NOTES)))]))
    return records


def ensure_demo(engine, url):
    """Remplit la table de démonstration la première fois qu'elle est vide."""
    if fetch_tests(url, TABLES["demo"]).empty:
        save_tests(engine, TABLES["demo"], demo_records())


# ============================================================
# 7. IMPORT CSV
# ============================================================

def parse_import(file_bytes, known_codes):
    raw = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding=encoding)
            break
        except Exception:
            continue
    if raw is None:
        return [], [t("e_file")]
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = {"sample_code", "test_date", "age_days", "specimen_type", "load_kn"} - set(raw.columns)
    if missing:
        return [], [t("e_cols", v=", ".join(sorted(missing)))]

    def value(row, column):
        v = row.get(column)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

    def number(row, column):
        v = value(row, column)
        return None if v is None or str(v).strip() == "" else float(str(v).replace(",", "."))

    records, errors, seen = [], [], set(known_codes)
    for index, row in raw.iterrows():
        line = t("line", n=index + 2)
        try:
            code = str(value(row, "sample_code") or "").strip()
            if code.lower() in seen:
                errors.append(f"{line} : {t('e_dup_line', v=code)}")
                continue
            specimen = str(value(row, "specimen_type") or "").strip().capitalize()
            specimen = {"Cylinder": "Cylindre"}.get(specimen, specimen)
            records.append(build_record(
                code, value(row, "project"), str(value(row, "test_date"))[:10], number(row, "age_days"), specimen,
                number(row, "diameter_mm"), number(row, "length_mm"), number(row, "width_mm"),
                number(row, "load_kn"), value(row, "notes")))
            seen.add(code.lower())
        except Exception as error:
            errors.append(f"{line} : {error}")
    return records, errors


# ============================================================
# 8. CONNEXION ET MENU
# ============================================================

DB_URL, PERMANENT = database_url()
try:
    engine = get_engine(DB_URL)
    init_database(DB_URL)
except Exception as err:
    get_engine.clear()
    init_database.clear()
    st.error(f"{t('conn_error')}\n\n{safe_error(err, DB_URL)}")
    st.stop()

PAGES = ["p_home", "p_new", "p_tests", "p_stats", "p_proba", "p_charts", "p_guide", "p_settings"]
ICONS = {"p_home": "🏠", "p_new": "➕", "p_tests": "🗂️", "p_stats": "📊", "p_proba": "🎯",
         "p_charts": "📈", "p_guide": "📘", "p_settings": "⚙️"}

with st.sidebar:
    st.markdown(f'<div class="bl-brand"><div class="bl-logo">🏗️</div><div><b>BetonLab</b>'
                f'<small>{E(t("tagline"))}</small></div></div>', unsafe_allow_html=True)
    st.write("")
    page = st.radio(t("nav"), PAGES, key="page", format_func=lambda k, L=lang(): f"{ICONS[k]}  {TR[L][k]}",
                    label_visibility="collapsed")
    st.divider()
    demo = st.toggle(t("demo_mode"), key="demo", help=t("demo_help"))
    st.segmented_control(t("language"), ["fr", "en"], key="lang", required=True, width="stretch",
                         format_func=lambda k: {"fr": "🇫🇷  Français", "en": "🇬🇧  English"}[k])
    st.segmented_control(t("theme"), ["light", "dark"], key="theme", required=True, width="stretch",
                         format_func=lambda k, L=lang(): ("☀️  " + TR[L]["light"]) if k == "light" else ("🌙  " + TR[L]["dark"]))
    st.query_params["lang"] = st.session_state["lang"]
    st.query_params["theme"] = st.session_state["theme"]
    st.divider()
    st.markdown(f'<div class="bl-foot">{E(t("footer"))}</div>', unsafe_allow_html=True)

TABLE = TABLES["demo" if demo else "real"]
show_flash()

try:
    if demo:
        ensure_demo(engine, DB_URL)
    df = fetch_tests(DB_URL, TABLE)
except Exception as err:
    fetch_tests.clear()
    st.error(f"{t('read_error')}\n\n{safe_error(err, DB_URL)}")
    st.stop()


def filter_ui(data, key):
    """Filtres chantier / type / âge communs aux pages d'analyse."""
    with st.expander(t("filters"), expanded=False, icon="🔎"):
        c1, c2, c3 = st.columns([2, 1, 1])
        projects = sorted(data["project"].unique())
        chosen = c1.multiselect(t("projects"), projects, default=projects, key=f"{key}_pr")
        types = sorted(data["specimen_type"].dropna().unique())
        options = ["__all"] + types
        default = types[0] if len(types) == 1 else "__all"
        chosen_type = c2.selectbox(t("c_type"), options, index=options.index(default), key=f"{key}_ty",
                                   format_func=lambda k, L=lang(): TR[L]["all"] if k == "__all" else TR[L][k])
        ages = sorted(int(a) for a in data["age_days"].dropna().unique())
        chosen_ages = c3.multiselect(t("ages"), ages, default=ages, key=f"{key}_ag")
    view = data[data["project"].isin(chosen) & data["age_days"].isin(chosen_ages)]
    if chosen_type != "__all":
        view = view[view["specimen_type"] == chosen_type]
    elif view["specimen_type"].nunique() > 1:
        st.caption("ℹ️ " + t("mixed"))
    return view


def empty_state():
    st.markdown(f'<div class="bl-card"><h3>{E(t("empty_title"))}</h3><p>{E(t("empty_text"))}</p></div>',
                unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.3, 1, 2])
    if not demo and c1.button(t("hero_demo"), type="primary", key=f"empty_demo_{page}"):
        st.session_state["_go_demo"] = True
        st.rerun()
    if c2.button(t("hero_new"), key=f"empty_new_{page}"):
        st.session_state["_go_page"] = "p_new"
        st.rerun()


# ============================================================
# 9. PAGES
# ============================================================

if page == "p_home":
    chip = (f'<span class="bl-chip">● {E(t("demo_chip"))}</span>' if demo else
            '<span class="bl-chip">🏗️ BetonLab</span>')
    st.markdown(f'<div class="bl-hero">{chip}<h1>{E(t("hero_title"))}</h1><p>{E(t("hero_text"))}</p></div>',
                unsafe_allow_html=True)
    buttons = st.columns([1.5, 1.2, 2.3])
    if not demo:
        if buttons[0].button("✨  " + t("hero_demo"), type="primary", key="hero_demo", width="stretch"):
            st.session_state["_go_demo"] = True
            st.rerun()
    if buttons[1 if not demo else 0].button("➕  " + t("hero_new"), key="hero_new", width="stretch",
                                            type="secondary" if not demo else "primary"):
        st.session_state["_go_page"] = "p_new"
        st.rerun()

    if df.empty:
        section(t("how_title"))
        steps = "".join(f'<div class="bl-step"><div class="n">{i}</div><b>{E(t(f"step{i}_t"))}</b>'
                        f'<span>{E(t(f"step{i}_d"))}</span></div>' for i in range(1, 5))
        st.markdown(f'<div class="bl-steps">{steps}</div>', unsafe_allow_html=True)
    else:
        s = df["strength_mpa"].dropna()
        kpis([
            (t("kpi_tests"), f"{len(df)}", "", t("kpi_tests_sub"), "primary"),
            (t("kpi_mean"), num(s.mean(), 1), "MPa", t("kpi_mean_sub"), "accent"),
            (t("kpi_min"), num(s.min(), 1), "MPa", "", ""),
            (t("kpi_max"), num(s.max(), 1), "MPa", "", ""),
            (t("kpi_projects"), f"{df['project'].nunique()}", "", "", ""),
        ])
        left, right = st.columns([1.35, 1])
        with left:
            section(t("trend"))
            data = df.dropna(subset=["test_date"]).copy()
            data["date"] = pd.to_datetime(data["test_date"])
            data["type"] = data["specimen_type"].map(t)
            chart = alt.Chart(data).mark_circle(size=70, opacity=.85).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("strength_mpa:Q", title="MPa", scale=alt.Scale(zero=False)),
                color=alt.Color("type:N", title=None, scale=alt.Scale(range=P["series"])),
                tooltip=[alt.Tooltip("sample_code:N", title=t("c_code")), alt.Tooltip("project:N", title=t("c_project")),
                         alt.Tooltip("age_days:Q", title=t("c_age")),
                         alt.Tooltip("strength_mpa:Q", title=t("c_strength"), format=".2f")])
            show_chart(chart, 300)
        with right:
            section(t("how_title"))
            steps = "".join(f'<div class="bl-step" style="padding:12px 14px"><b>{i}. {E(t(f"step{i}_t"))}</b>'
                            f'<span>{E(t(f"step{i}_d"))}</span></div>' for i in range(1, 5))
            st.markdown(f'<div style="display:grid;gap:10px">{steps}</div>', unsafe_allow_html=True)
        section(t("recent"))
        tests_table(df, limit=8)


elif page == "p_new":
    page_header(t("new_title"), t("new_sub"))
    specimen = st.segmented_control(t("specimen"), SPECIMEN_TYPES, key="specimen", required=True,
                                    default="Cube", format_func=lambda k, L=lang(): TR[L][k])
    form_col, result_col = st.columns([1.6, 1])
    with form_col:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            code = c1.text_input(t("code"), placeholder="BET-001", max_chars=100)
            project = c2.text_input(t("project"), value=t("project_default"), max_chars=200)
            test_date = c1.date_input(t("test_date"), value=date.today(), max_value=date.today(),
                                      format="DD/MM/YYYY" if lang() == "fr" else "YYYY-MM-DD")
            age = c2.number_input(t("age"), min_value=0, max_value=3650, value=28, step=1)
            diameter = length = width = None
            if specimen == "Cylindre":
                diameter = c1.number_input(t("diameter"), min_value=1.0, value=160.0, step=1.0)
            else:
                length = c1.number_input(t("length"), min_value=1.0, value=150.0, step=1.0)
                width = c2.number_input(t("width"), min_value=1.0, value=150.0, step=1.0)
            load = c2.number_input(t("load"), min_value=0.0, value=750.0, step=1.0) if specimen == "Cylindre" \
                else c1.number_input(t("load"), min_value=0.0, value=750.0, step=1.0)
            notes = st.text_area(t("notes"), height=80)
            submitted = st.button(t("save"), type="primary", width="stretch", key="btn_save")
    with result_col:
        try:
            area = calculate_area(specimen, diameter, length, width)
            preview = load * 1000 / area if load and area else None
        except ValueError:
            area, preview = None, None
        st.markdown(f'<div class="bl-result"><div class="lbl">{E(t("preview"))}</div>'
                    f'<div class="val">{E(num(preview, 2))} <span style="font-size:1rem">MPa</span></div>'
                    f'<div class="sub">{E(t("area"))} : {E(num(area, 0))} mm² · F = {E(num(load, 1))} kN</div></div>',
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="bl-card"><p>f = F × 1000 / A<br>'
                    f'{"A = π·d²/4" if specimen == "Cylindre" else "A = a × b"}</p></div>', unsafe_allow_html=True)
    if submitted:
        try:
            record = build_record(code, project, test_date, age, specimen, diameter, length, width, load, notes)
            if record["sample_code"].lower() in existing_codes(engine, TABLE):
                raise ValueError(t("e_dup", v=record["sample_code"]))
            save_tests(engine, TABLE, [record])
            flash("success", f"{t('saved')} {record['sample_code']} : {num(record['strength_mpa'], 2)} MPa")
            st.rerun()
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"{t('write_error')} {safe_error(error, DB_URL)}")


elif page == "p_tests":
    page_header(t("tests_title"), t("tests_sub"))
    if df.empty:
        empty_state()
    else:
        c1, c2 = st.columns([2, 1.2])
        search = c1.text_input(t("search"), key="search", placeholder="BET-…")
        dates = df["test_date"].dropna()
        period = c2.date_input(t("period"), value=(min(dates), max(dates)), key="period",
                               format="DD/MM/YYYY" if lang() == "fr" else "YYYY-MM-DD")
        view = df
        if search.strip():
            q = search.strip()
            view = view[view["sample_code"].astype(str).str.contains(q, case=False, regex=False)
                        | view["project"].astype(str).str.contains(q, case=False, regex=False)
                        | view["notes"].astype(str).str.contains(q, case=False, regex=False)]
        if isinstance(period, (tuple, list)) and len(period) == 2:
            view = view[view["test_date"].notna() & (view["test_date"] >= period[0]) & (view["test_date"] <= period[1])]
        st.caption(t("shown", n=len(view), total=len(df)))
        tests_table(view)
        st.write("")
        e1, _ = st.columns([1, 3])
        e1.download_button("⬇️  " + t("export"), view.drop(columns=["created_at"]).to_csv(index=False, sep=";").encode("utf-8-sig"),
                           file_name=f"{t('dl_name')}_{date.today().isoformat()}.csv", mime="text/csv", key="export")
        with st.expander(t("delete_sel"), icon="🗑️"):
            labels = {int(r.id): f"{r.sample_code} · {r.project} · {num(r.strength_mpa, 1)} MPa" for r in view.itertuples()}
            chosen = st.multiselect(t("delete_pick"), list(labels), format_func=labels.get, key="to_delete")
            if st.button(t("delete_btn"), disabled=not chosen, key="btn_delete"):
                try:
                    n = delete_tests(engine, TABLE, chosen)
                    st.session_state.pop("to_delete", None)
                    flash("success", t("deleted", n=n))
                    st.rerun()
                except Exception as error:
                    st.error(f"{t('write_error')} {safe_error(error, DB_URL)}")


elif page == "p_stats":
    page_header(t("stats_title"), t("stats_sub"))
    if df.empty:
        empty_state()
    else:
        view = filter_ui(df, "st")
        s = view["strength_mpa"].dropna()
        if s.empty:
            st.info(t("no_match"))
        else:
            sd = s.std(ddof=1) if len(s) > 1 else float("nan")
            cv = sd / s.mean() * 100 if len(s) > 1 and s.mean() else float("nan")
            kpis([
                (t("s_n"), f"{len(s)}", "", "", "primary"),
                (t("s_mean"), num(s.mean(), 2), "MPa", "", "accent"),
                (t("s_median"), num(s.median(), 2), "MPa", "", ""),
                (t("s_sd"), num(sd, 2), "MPa", "", ""),
                (t("s_cv"), num(cv, 1), "%", "", ""),
                (t("s_min"), num(s.min(), 2), "MPa", "", ""),
                (t("s_max"), num(s.max(), 2), "MPa", "", ""),
            ])
            st.caption("ℹ️ " + t("help_stats"))
            def grouped(by):
                g = view.groupby(by)["strength_mpa"].agg(["count", "mean", "std", "min", "max"]).reset_index()
                return [[(f"{int(r[by])} j" if lang() == "fr" else f"{int(r[by])} d") if by == "age_days" else r[by],
                         int(r["count"]), num(r["mean"], 2), num(r["std"], 2), num(r["min"], 2), num(r["max"], 2)]
                        for _, r in g.iterrows()]

            a, b = st.columns(2)
            with a:
                section(t("by_age"))
                simple_table([t("c_age"), t("s_n"), t("s_mean"), t("s_sd"), t("s_min"), t("s_max")], grouped("age_days"))
            with b:
                section(t("by_project"))
                simple_table([t("c_project"), t("s_n"), t("s_mean"), t("s_sd"), t("s_min"), t("s_max")], grouped("project"))


elif page == "p_proba":
    page_header(t("proba_title"), t("proba_sub"))
    if df.empty:
        empty_state()
    else:
        view = filter_ui(df, "pr")
        values = view["strength_mpa"].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            st.info(t("need2"))
        elif float(np.std(values, ddof=1)) <= 0:
            st.info(t("sd0"))
        else:
            m, s = float(values.mean()), float(np.std(values, ddof=1))
            fk = m - 1.645 * s
            c1, c2 = st.columns([1, 1.6])
            with c1:
                threshold = st.number_input(t("threshold"), min_value=0.0, value=float(round(fk, 1)), step=0.5, key="thr")
                below = float(norm.cdf(threshold, loc=m, scale=s))
                kpis([(t("s_mean"), num(m, 2), "MPa", "", "primary"), (t("s_sd"), num(s, 2), "MPa", "", ""),
                      (t("fk"), num(fk, 2), "MPa", t("fk_help"), "accent")])
                kpis([(t("p_above"), num((1 - below) * 100, 1), "%", "", ""),
                      (t("p_below"), num(below * 100, 1), "%", "", "")])
            with c2:
                bins = min(16, max(5, int(np.sqrt(len(values))) + 2))
                counts, edges = np.histogram(values, bins=bins, density=True)
                hist = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "d": counts})
                xs = np.linspace(min(values.min(), m - 4 * s), max(values.max(), m + 4 * s), 200)
                curve = pd.DataFrame({"x": xs, "d": norm.pdf(xs, m, s)})
                bars = alt.Chart(hist).mark_rect(color=P["series"][0], opacity=.45, stroke=P["surface"],
                                                 strokeWidth=1).encode(
                    x=alt.X("lo:Q", title="MPa"), x2="hi:Q", y=alt.Y("d:Q", title=t("density")), y2=alt.datum(0))
                line = alt.Chart(curve).mark_line(color=P["text"], strokeWidth=2.2).encode(x="x:Q", y="d:Q")
                rules = alt.Chart(pd.DataFrame({"x": [threshold, fk], "c": [t("threshold"), t("fk")]})).mark_rule(
                    strokeDash=[6, 4], strokeWidth=2).encode(
                    x="x:Q", color=alt.Color("c:N", title=None, scale=alt.Scale(range=[P["bad"], P["accent"]])))
                show_chart(bars + line + rules, 340)
            if len(values) < 15:
                st.caption("⚠️ " + t("few", n=len(values)))
            st.caption("ℹ️ " + t("proba_note"))


elif page == "p_charts":
    page_header(t("charts_title"), t("charts_sub"))
    if df.empty:
        empty_state()
    else:
        view = filter_ui(df, "ch")
        if view["strength_mpa"].dropna().empty:
            st.info(t("no_match"))
        else:
            data = view.copy()
            data["type"] = data["specimen_type"].map(t)
            mean = float(data["strength_mpa"].mean())
            a, b = st.columns(2)
            with a:
                section(t("g_hist"))
                hist = alt.Chart(data).mark_bar(color=P["series"][0], cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("strength_mpa:Q", bin=alt.Bin(maxbins=14), title="MPa"),
                    y=alt.Y("count():Q", title=t("count")))
                rule = alt.Chart(pd.DataFrame({"m": [mean]})).mark_rule(color=P["accent"], strokeDash=[6, 4], strokeWidth=2).encode(
                    x="m:Q", tooltip=[alt.Tooltip("m:Q", title=t("mean_line"), format=".2f")])
                show_chart(hist + rule, 300)
            with b:
                section(t("g_age"))
                means = data.groupby("age_days")["strength_mpa"].mean().reset_index()
                pts = alt.Chart(data).mark_circle(size=60, opacity=.55).encode(
                    x=alt.X("age_days:Q", title=t("c_age"), scale=alt.Scale(type="sqrt")),
                    y=alt.Y("strength_mpa:Q", title="MPa", scale=alt.Scale(zero=False)),
                    color=alt.Color("type:N", title=None, scale=alt.Scale(range=P["series"])),
                    tooltip=[alt.Tooltip("sample_code:N", title=t("c_code")), alt.Tooltip("age_days:Q", title=t("c_age")),
                             alt.Tooltip("strength_mpa:Q", title="MPa", format=".2f")])
                line = alt.Chart(means).mark_line(color=P["text"], point=True, strokeWidth=2).encode(
                    x="age_days:Q", y="strength_mpa:Q")
                show_chart(pts + line, 300)
            section(t("g_project"))
            box = alt.Chart(data).mark_boxplot(size=34, color=P["series"][0], median={"color": P["accent"]},
                                               rule={"color": P["muted"]}, outliers={"color": P["bad"]}).encode(
                x=alt.X("project:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=180)),
                y=alt.Y("strength_mpa:Q", title="MPa", scale=alt.Scale(zero=False)))
            show_chart(box, 300)
            section(t("g_last"))
            last = data.sort_values(["test_date", "id"]).tail(30)
            bars = alt.Chart(last).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("sample_code:N", sort=None, title=None, axis=alt.Axis(labelAngle=-50)),
                y=alt.Y("strength_mpa:Q", title="MPa"),
                color=alt.Color("type:N", title=None, scale=alt.Scale(range=P["series"])),
                tooltip=[alt.Tooltip("sample_code:N", title=t("c_code")), alt.Tooltip("project:N", title=t("c_project")),
                         alt.Tooltip("strength_mpa:Q", title="MPa", format=".2f")])
            show_chart(bars, 300)


elif page == "p_guide":
    page_header(t("guide_title"))
    st.markdown(f'<div class="bl-card"><p style="color:var(--text)!important;font-size:1.02rem">{E(t("guide_intro"))}</p></div>',
                unsafe_allow_html=True)
    cards = "".join(f'<div class="bl-card"><h3>{E(t(f"g{i}_t"))}</h3><p>{E(t(f"g{i}_d"))}</p></div>' for i in range(1, 7))
    st.markdown(f'<div class="bl-guide">{cards}</div>', unsafe_allow_html=True)
    if not demo and st.button(t("hero_demo"), type="primary", key="guide_demo"):
        st.session_state["_go_demo"] = True
        st.rerun()


elif page == "p_settings":
    page_header(t("settings_title"), t("settings_sub"))
    section(t("db"))
    if PERMANENT:
        st.success(t("db_ok"), icon="✅")
    else:
        st.warning(t("db_temp"), icon="⚠️")

    section(t("demo_box"))
    if demo:
        if st.button(t("demo_reload"), key="btn_demo_reload"):
            try:
                delete_all(engine, TABLES["demo"])
                n = save_tests(engine, TABLES["demo"], demo_records())
                flash("success", t("demo_reloaded", n=n))
                st.rerun()
            except Exception as error:
                st.error(f"{t('write_error')} {safe_error(error, DB_URL)}")
    else:
        st.caption(t("demo_only"))

    section(t("import"))
    st.caption(t("import_help"))
    uploaded = st.file_uploader(t("file"), type=["csv"], key="upload")
    if uploaded is not None:
        records, errors = parse_import(uploaded.getvalue(), existing_codes(engine, TABLE))
        st.write(t("valid_rows", ok=len(records), bad=len(errors)))
        if errors:
            with st.expander(t("see_problems")):
                for e in errors[:200]:
                    st.write("• " + e)
        if records and st.button(t("import_btn", n=len(records)), type="primary", key="btn_import"):
            try:
                n = save_tests(engine, TABLE, records)
                st.session_state.pop("upload", None)
                flash("success", t("imported", n=n))
                st.rerun()
            except Exception as error:
                st.error(f"{t('write_error')} {safe_error(error, DB_URL)}")

    section(t("danger"))
    with st.expander(t("delete_all"), icon="⚠️"):
        ok = st.checkbox(t("delete_all_confirm"), key="confirm_delete")
        word = "SUPPRIMER" if lang() == "fr" else "DELETE"
        phrase = st.text_input(t("type_delete"), key="confirm_phrase")
        if st.button(t("delete_all"), key="btn_delete_all"):
            if ok and phrase.strip().upper() == word:
                try:
                    delete_all(engine, TABLE)
                    for k in ("confirm_delete", "confirm_phrase"):
                        st.session_state.pop(k, None)
                    flash("success", t("deleted_all"))
                    st.rerun()
                except Exception as error:
                    st.error(f"{t('write_error')} {safe_error(error, DB_URL)}")
            else:
                st.error(t("confirm_needed"))
