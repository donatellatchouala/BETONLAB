BetonLab — Analyse des essais de résistance du béton

import os
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

# =========================================================
# 1. CONFIGURATION GENERALE
# =========================================================

st.set_page_config(
    page_title="BetonLab | Analyse du béton",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "betonlab.db")

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
        "graphique": "plotly_white",
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
        "graphique": "plotly_dark",
    }
}

if "theme_betonlab" not in st.session_state:
    st.session_state.theme_betonlab = "Clair"

# =========================================================
# 3. BASE DE DONNEES
# =========================================================

def connexion():
    return sqlite3.connect(DB_PATH, timeout=15)


def initialiser_base():
    with connexion() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_code TEXT NOT NULL,
                test_date TEXT NOT NULL,
                age_days INTEGER NOT NULL,
                length_mm REAL NOT NULL,
                width_mm REAL NOT NULL,
                height_mm REAL NOT NULL,
                load_kn REAL NOT NULL,
                strength_mpa REAL NOT NULL
            )
        """)

        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_tests_date
            ON tests(test_date)
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS historique (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_heure TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT DEFAULT ''
            )
        """)

        # Migration pour les anciennes bases BetonLab.
        # La nouvelle colonne conserve la forme de l'éprouvette.
        colonnes = [
            ligne[1]
            for ligne in con.execute(
                "PRAGMA table_info(tests)"
            ).fetchall()
        ]

        if "shape" not in colonnes:
            con.execute("""
                ALTER TABLE tests
                ADD COLUMN shape TEXT DEFAULT 'Prismatique'
            """)

        con.commit()


def enregistrer_action(action, details=""):
    with connexion() as con:
        con.execute("""
            INSERT INTO historique(date_heure, action, details)
            VALUES (?, ?, ?)
        """, (
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            action,
            details
        ))


@st.cache_data(ttl=20, show_spinner=False)
def charger_essais():
    with connexion() as con:
        df = pd.read_sql_query(
            """
            SELECT * FROM tests
            ORDER BY test_date DESC, id DESC
            """,
            con
        )

    if not df.empty:
        df["test_date"] = pd.to_datetime(
            df["test_date"],
            errors="coerce"
        )

    return df


def charger_historique():
    with connexion() as con:
        return pd.read_sql_query("""
            SELECT
                date_heure AS 'Date et heure',
                action AS 'Action',
                details AS 'Détails'
            FROM historique
            ORDER BY id DESC
        """, con)


def sauvegarder_essai(
    code, jour, age, longueur, largeur, hauteur,
    charge, resistance, forme
):
    with connexion() as con:
        con.execute("""
            INSERT INTO tests (
                sample_code, test_date, age_days,
                length_mm, width_mm, height_mm,
                load_kn, strength_mpa, shape
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            code, jour, age, longueur, largeur, hauteur,
            charge, resistance, forme
        ))

    charger_essais.clear()

    enregistrer_action(
        "Ajout d'un essai",
        f"Éprouvette {code} — résistance : {resistance:.2f} MPa"
    )


def supprimer_essai(identifiant):
    with connexion() as con:
        ligne = con.execute(
            "SELECT sample_code FROM tests WHERE id = ?",
            (identifiant,)
        ).fetchone()

        if ligne:
            con.execute(
                "DELETE FROM tests WHERE id = ?",
                (identifiant,)
            )

    if ligne:
        charger_essais.clear()
        enregistrer_action(
            "Suppression d'un essai",
            f"Identifiant {identifiant} — éprouvette {ligne[0]}"
        )

    return ligne is not None


initialiser_base()

# =========================================================
# 4. CALCUL DE LA RESISTANCE
# =========================================================

def calculer_resistance(charge_kn, longueur_mm, largeur_mm, forme):
    """
    La charge est exprimée en kN.
    Les dimensions sont exprimées en mm.
    La résistance obtenue est exprimée en MPa.
    """

    if charge_kn <= 0:
        raise ValueError("La charge doit être positive.")

    if longueur_mm <= 0 or largeur_mm <= 0:
        raise ValueError("Les dimensions doivent être positives.")

    if forme == "Cylindrique":
        surface = np.pi * longueur_mm**2 / 4
    else:
        surface = longueur_mm * largeur_mm

    return charge_kn * 1000 / surface


# =========================================================
# 5. STATISTIQUES
# =========================================================

def calculer_statistiques(df):
    x = df["strength_mpa"].dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return {}

    moyenne = float(np.mean(x))
    ecart_type = (
        float(np.std(x, ddof=1))
        if len(x) > 1 else None
    )

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
# 6. STYLE ET GRAPHIQUES
# =========================================================

def appliquer_style_graphique(fig, titre):
    couleurs = THEMES[st.session_state.theme_betonlab]

    fig.update_layout(
        title=titre,
        template=couleurs["graphique"],
        font=dict(
            family="Arial",
            size=14,
            color=couleurs["texte"]
        ),
        title_font=dict(
            size=20,
            color=couleurs["primaire"]
        ),
        margin=dict(l=25, r=25, t=65, b=35),
        hovermode="closest",
        paper_bgcolor=couleurs["fond"],
        plot_bgcolor=couleurs["fond"],
        height=400
    )

    return fig


def graphique_histogramme(df):
    couleurs = THEMES[st.session_state.theme_betonlab]

    fig = px.histogram(
        df,
        x="strength_mpa",
        nbins=max(5, min(20, len(df))),
        color_discrete_sequence=[couleurs["accent"]],
        labels={
            "strength_mpa": "Résistance (MPa)",
            "count": "Nombre d'essais"
        }
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color=couleurs["fond"]
    )

    return appliquer_style_graphique(
        fig,
        "Répartition des résistances"
    )


def graphique_evolution(df):
    couleurs = THEMES[st.session_state.theme_betonlab]

    ordre = df.sort_values("test_date").copy()

    fig = px.line(
        ordre,
        x="test_date",
        y="strength_mpa",
        markers=True,
        color_discrete_sequence=[couleurs["accent"]],
        labels={
            "test_date": "Date de l'essai",
            "strength_mpa": "Résistance (MPa)"
        },
        hover_data=["sample_code", "age_days"]
    )

    return appliquer_style_graphique(
        fig,
        "Évolution de la résistance dans le temps"
    )


def graphique_boite(df):
    couleurs = THEMES[st.session_state.theme_betonlab]

    fig = px.box(
        df,
        y="strength_mpa",
        points="all",
        color_discrete_sequence=[couleurs["accent"]],
        labels={
            "strength_mpa": "Résistance (MPa)"
        }
    )

    return appliquer_style_graphique(
        fig,
        "Dispersion des résistances"
    )


def graphique_normale(df):
    couleurs = THEMES[st.session_state.theme_betonlab]
    x = df["strength_mpa"].dropna().to_numpy(dtype=float)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=x,
        histnorm="probability density",
        nbinsx=max(5, min(20, len(x))),
        name="Résultats observés",
        marker_color=couleurs["primaire"],
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

            yy = stats.norm.pdf(
                xx, loc=moyenne, scale=ecart_type
            )

            fig.add_trace(go.Scatter(
                x=xx,
                y=yy,
                mode="lines",
                name="Modèle normal",
                line=dict(
                    color=couleurs["accent"],
                    width=3
                )
            ))

    fig.update_layout(
        xaxis_title="Résistance (MPa)",
        yaxis_title="Densité de probabilité"
    )

    return appliquer_style_graphique(
        fig,
        "Résultats observés et modèle normal"
    )


# =========================================================
# 7. APPARENCE DE L'APPLICATION
# =========================================================

with st.sidebar:
    st.title("⚙️ Paramètres")

    theme = st.radio(
        "Choisir l'apparence",
        ["Clair", "Sombre"],
        index=(
            0 if st.session_state.theme_betonlab == "Clair"
            else 1
        ),
        key="choix_theme"
    )

    if theme != st.session_state.theme_betonlab:
        st.session_state.theme_betonlab = theme
        st.rerun()

C = THEMES[st.session_state.theme_betonlab]

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {C["fond"]};
        color: {C["texte"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {C["panneau"]};
        border-right: 1px solid {C["bordure"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {C["texte"]};
    }}

    h1, h2, h3, h4, p, label {{
        color: {C["texte"]};
    }}

    [data-testid="stMetric"] {{
        background-color: {C["panneau"]};
        border: 1px solid {C["bordure"]};
        padding: 16px;
        border-radius: 12px;
    }}

    div.stButton > button,
    div.stDownloadButton > button {{
        border-radius: 9px;
        border: 1px solid {C["bordure"]};
        font-weight: 600;
    }}

    .hero {{
        background: linear-gradient(
            110deg, {C["primaire"]}, {C["accent"]}
        );
        padding: 26px;
        border-radius: 16px;
        margin-bottom: 20px;
    }}

    .hero h1, .hero p {{
        color: white !important;
    }}

    .hero h1 {{
        margin: 0;
        font-size: 2.1rem;
    }}

    .hero p {{
        margin-bottom: 0;
        font-size: 1rem;
    }}

    .carte {{
        background-color: {C["panneau"]};
        border: 1px solid {C["bordure"]};
        padding: 17px;
        border-radius: 12px;
        min-height: 100px;
    }}

    .carte-titre {{
        color: {C["secondaire"]} !important;
        font-size: .9rem;
    }}

    .carte-valeur {{
        color: {C["texte"]} !important;
        font-size: 1.55rem;
        font-weight: 750;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


def afficher_carte(titre, valeur, unite=""):
    st.markdown(
        f"""
        <div class="carte">
            <div class="carte-titre">{titre}</div>
            <div class="carte-valeur">{valeur} {unite}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 8. TITRE ET NAVIGATION
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🏗️ BetonLab</h1>
    <p>Application d'analyse des essais de résistance à la compression du béton</p>
    <p>Statistiques • Probabilités • Diagrammes • Historique</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("BETONLAB")
    st.caption("Laboratoire numérique du béton")

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

    st.divider()
    st.caption("Version 2.0")
    st.caption("Python • Streamlit • Plotly")


# Enregistrer une consultation seulement lors d'un changement
# de page dans la session, afin d'éviter les doublons à chaque rerun.
if st.session_state.get("derniere_page") != page:
    st.session_state.derniere_page = page
    enregistrer_action("Consultation d'une rubrique", page)

df = charger_essais()
s = calculer_statistiques(df)


# =========================================================
# 9. TABLEAU DE BORD
# =========================================================

if page == "Tableau de bord":

    st.header("📊 Vue d'ensemble du laboratoire")
    st.write(
        "Retrouvez les principaux indicateurs, les derniers essais "
        "et les tendances de résistance de vos éprouvettes."
    )

    if df.empty:
        st.info(
            "Aucun essai enregistré. Ouvrez « Enregistrer un essai » "
            "pour commencer à constituer votre base de données."
        )

        a, b, c = st.columns(3)

        with a:
            st.subheader("🧪 Saisir")
            st.write("Enregistrez les dimensions, l'âge et la charge de rupture.")

        with b:
            st.subheader("📈 Analyser")
            st.write("Consultez les statistiques, les probabilités et les diagrammes.")

        with c:
            st.subheader("🗂️ Conserver")
            st.write("Retrouvez les opérations dans l'historique d'utilisation.")

    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            afficher_carte("Nombre d'essais", str(s["nombre"]))

        with c2:
            afficher_carte(
                "Résistance moyenne",
                nombre_francais(s["moyenne"]), "MPa"
            )

        with c3:
            afficher_carte(
                "Résistance maximale",
                nombre_francais(s["maximum"]), "MPa"
            )

        with c4:
            afficher_carte(
                "Résistance minimale",
                nombre_francais(s["minimum"]), "MPa"
            )

        st.write("")

        c5, c6, c7, c8 = st.columns(4)

        with c5:
            afficher_carte(
                "Médiane",
                nombre_francais(s["mediane"]), "MPa"
            )

        with c6:
            afficher_carte(
                "Étendue",
                nombre_francais(s["etendue"]), "MPa"
            )

        with c7:
            afficher_carte(
                "Écart-type",
                nombre_francais(s["ecart_type"]), "MPa"
            )

        with c8:
            afficher_carte(
                "Coefficient de variation",
                nombre_francais(s["cv"]), "%"
            )

        st.write("")
        gauche, droite = st.columns(2)

        with gauche:
            st.plotly_chart(
                graphique_histogramme(df),
                use_container_width=True
            )

        with droite:
            if len(df) >= 2:
                st.plotly_chart(
                    graphique_evolution(df),
                    use_container_width=True
                )
            else:
                st.info(
                    "Ajoutez plusieurs essais pour visualiser "
                    "l'évolution de la résistance."
                )

        st.subheader("🧾 Derniers essais enregistrés")

        derniers = df.head(8).copy()

        derniers["test_date"] = (
            derniers["test_date"].dt.strftime("%d/%m/%Y")
        )

        derniers = derniers.rename(columns={
            "id": "Identifiant",
            "sample_code": "Code éprouvette",
            "test_date": "Date",
            "age_days": "Âge (jours)",
            "load_kn": "Charge (kN)",
            "strength_mpa": "Résistance (MPa)",
            "shape": "Forme"
        })

        colonnes = [
            "Identifiant", "Code éprouvette", "Date",
            "Âge (jours)", "Charge (kN)",
            "Résistance (MPa)", "Forme"
        ]

        st.dataframe(
            derniers[[c for c in colonnes if c in derniers.columns]],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🧭 Aperçu des analyses")

        a, b, c = st.columns(3)

        with a:
            st.markdown("**Statistiques descriptives**")
            st.write(
                "Moyenne, médiane, dispersion et coefficient de variation."
            )

        with b:
            st.markdown("**Probabilités**")
            st.write(
                "Estimation de la probabilité d'atteindre une résistance cible."
            )

        with c:
            st.markdown("**Suivi des opérations**")
            st.write(
                "Historique des ajouts, suppressions, exports et consultations."
            )


# =========================================================
# 10. ENREGISTRER UN ESSAI
# =========================================================

elif page == "Enregistrer un essai":

    st.header("🧪 Enregistrer un essai de compression")

    st.write(
        "Remplissez les informations de l'éprouvette. "
        "La résistance sera calculée automatiquement."
    )

    with st.form("formulaire_essai", clear_on_submit=True):

        col1, col2 = st.columns(2)

        with col1:
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

        with col2:
            if forme == "Cylindrique":
                longueur = st.number_input(
                    "Diamètre (mm)",
                    min_value=0.1,
                    value=150.0
                )

                largeur = longueur

                st.caption(
                    "Pour un cylindre, le calcul utilise la surface "
                    "circulaire π × diamètre² / 4."
                )

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
                "Hauteur de l'éprouvette (mm)",
                min_value=0.1,
                value=300.0
            )

            charge = st.number_input(
                "Charge maximale de rupture (kN)",
                min_value=0.1,
                value=650.0
            )

        envoyer = st.form_submit_button(
            "Calculer et enregistrer",
            use_container_width=True
        )

    if envoyer:
        if not code.strip():
            st.error("Veuillez saisir le code de l'éprouvette.")

        else:
            resistance = calculer_resistance(
                charge, longueur, largeur, forme
            )

            sauvegarder_essai(
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

            st.success("L'essai a été enregistré avec succès.")

            st.metric(
                "Résistance calculée",
                f"{resistance:.2f} MPa"
            )

            st.info(
                "Pour une éprouvette prismatique : "
                "résistance = charge (N) / surface (mm²). "
                "Pour une éprouvette cylindrique, la surface est "
                "π × diamètre² / 4."
            )

            st.rerun()


# =========================================================
# 11. BASE DES ESSAIS
# =========================================================

elif page == "Base des essais":

    st.header("🗂️ Base de données des essais")

    if df.empty:
        st.info("La base de données est vide.")

    else:
        recherche = st.text_input(
            "Rechercher par code d'éprouvette"
        )

        vue = df.copy()

        if recherche.strip():
            vue = vue[
                vue["sample_code"].str.contains(
                    recherche.strip(),
                    case=False,
                    na=False
                )
            ]

        affichage = vue.rename(columns={
            "id": "Identifiant",
            "sample_code": "Code éprouvette",
            "test_date": "Date de l'essai",
            "age_days": "Âge (jours)",
            "length_mm": "Longueur ou diamètre (mm)",
            "width_mm": "Largeur (mm)",
            "height_mm": "Hauteur (mm)",
            "load_kn": "Charge (kN)",
            "strength_mpa": "Résistance (MPa)",
            "shape": "Forme"
        })

        if "Date de l'essai" in affichage.columns:
            affichage["Date de l'essai"] = pd.to_datetime(
                affichage["Date de l'essai"]
            ).dt.strftime("%d/%m/%Y")

        st.dataframe(
            affichage,
            use_container_width=True,
            hide_index=True
        )

        csv_data = vue.to_csv(
            index=False
        ).encode("utf-8-sig")

        if st.download_button(
            "⬇️ Exporter les résultats en CSV",
            data=csv_data,
            file_name="betonlab_essais.csv",
            mime="text/csv"
        ):
            enregistrer_action(
                "Exportation des essais",
                f"{len(vue)} enregistrement(s) exporté(s)"
            )

        st.divider()
        st.subheader("Supprimer un essai")

        identifiants = vue["id"].astype(int).tolist()

        if identifiants:
            identifiant = st.selectbox(
                "Choisir l'essai",
                identifiants,
                format_func=lambda i: (
                    f"ID {i} — "
                    f"{vue.loc[vue['id'] == i, 'sample_code'].iloc[0]}"
                )
            )

            confirmation = st.checkbox(
                "Je confirme vouloir supprimer définitivement cet essai."
            )

            if st.button("Supprimer l'essai sélectionné"):
                if confirmation:
                    if supprimer_essai(int(identifiant)):
                        st.success("L'essai a été supprimé.")
                        st.rerun()
                else:
                    st.warning(
                        "Veuillez confirmer la suppression avant de continuer."
                    )


# =========================================================
# 12. STATISTIQUES
# =========================================================

elif page == "Statistiques":

    st.header("📊 Analyse statistique")

    if s == {}:
        st.info(
            "Enregistrez des essais pour afficher les statistiques."
        )

    else:
        colonnes = st.columns(4)

        indicateurs = [
            ("Nombre d'essais", str(s["nombre"])),
            ("Moyenne", f'{s["moyenne"]:.2f} MPa'),
            ("Médiane", f'{s["mediane"]:.2f} MPa'),
            ("Étendue", f'{s["etendue"]:.2f} MPa')
        ]

        for col, (titre, valeur) in zip(colonnes, indicateurs):
            with col:
                st.metric(titre, valeur)

        colonnes = st.columns(4)

        indicateurs = [
            (
                "Minimum",
                f'{s["minimum"]:.2f} MPa'
            ),
            (
                "Maximum",
                f'{s["maximum"]:.2f} MPa'
            ),
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

        for col, (titre, valeur) in zip(colonnes, indicateurs):
            with col:
                st.metric(titre, valeur)

        st.subheader("Tableau récapitulatif")

        resume = pd.DataFrame({
            "Indicateur": [
                "Nombre d'essais",
                "Moyenne",
                "Médiane",
                "Minimum",
                "Maximum",
                "Étendue",
                "Écart-type",
                "Coefficient de variation"
            ],
            "Valeur": [
                s["nombre"],
                s["moyenne"],
                s["mediane"],
                s["minimum"],
                s["maximum"],
                s["etendue"],
                s["ecart_type"],
                s["cv"]
            ]
        })

        st.dataframe(
            resume,
            use_container_width=True,
            hide_index=True
        )

        gauche, droite = st.columns(2)

        with gauche:
            st.plotly_chart(
                graphique_boite(df),
                use_container_width=True
            )

        with droite:
            st.plotly_chart(
                graphique_histogramme(df),
                use_container_width=True
            )

        st.subheader("Test indicatif de normalité")

        valeurs = df["strength_mpa"].dropna().to_numpy(dtype=float)

        if len(valeurs) >= 3 and np.std(valeurs) > 0:
            echantillon = (
                valeurs if len(valeurs) <= 5000
                else np.random.default_rng(42).choice(
                    valeurs, size=5000, replace=False
                )
            )

            resultat = stats.shapiro(echantillon)

            st.write(
                f"Statistique de Shapiro-Wilk : "
                f"**{resultat.statistic:.4f}**"
            )

            st.write(
                f"Valeur p : **{resultat.pvalue:.4f}**"
            )

            if resultat.pvalue < 0.05:
                st.warning(
                    "Au seuil de 5 %, le test fournit des éléments "
                    "contre l'hypothèse de normalité."
                )
            else:
                st.info(
                    "Au seuil de 5 %, le test ne permet pas de rejeter "
                    "l'hypothèse de normalité. Cela ne prouve pas "
                    "que les données suivent une loi normale."
                )

        else:
            st.info(
                "Le test nécessite au moins trois valeurs et une "
                "dispersion non nulle."
            )


# =========================================================
# 13. PROBABILITES
# =========================================================

elif page == "Probabilités":

    st.header("🎯 Analyse des probabilités")

    st.write(
        "Cette rubrique estime les probabilités à partir d'un "
        "modèle normal ajusté aux résistances observées."
    )

    valeurs = df["strength_mpa"].dropna().to_numpy(dtype=float)

    if len(valeurs) < 2:
        st.warning(
            "Enregistrez au moins deux essais pour calculer "
            "les probabilités."
        )

    else:
        moyenne = float(np.mean(valeurs))
        ecart_type = float(np.std(valeurs, ddof=1))

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Résistance moyenne",
                f"{moyenne:.2f} MPa"
            )

        with c2:
            st.metric(
                "Écart-type",
                f"{ecart_type:.2f} MPa"
            )

        cible = st.number_input(
            "Résistance cible (MPa)",
            min_value=0.0,
            value=float(round(moyenne, 1)),
            step=0.5
        )

        if ecart_type == 0:
            st.warning(
                "La dispersion est nulle. Un modèle normal avec "
                "écart-type nul ne peut pas être ajusté."
            )

        else:
            p_inferieur = float(
                stats.norm.cdf(
                    cible,
                    loc=moyenne,
                    scale=ecart_type
                )
            )

            p_superieur = 1 - p_inferieur

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    f"P(X ≤ {cible:.2f} MPa)",
                    f"{p_inferieur * 100:.2f}%"
                )

            with c2:
                st.metric(
                    f"P(X ≥ {cible:.2f} MPa)",
                    f"{p_superieur * 100:.2f}%"
                )

            couleurs = THEMES[st.session_state.theme_betonlab]

            xx = np.linspace(
                min(0, moyenne - 4 * ecart_type),
                moyenne + 4 * ecart_type,
                400
            )

            yy = stats.norm.pdf(
                xx,
                loc=moyenne,
                scale=ecart_type
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=xx,
                y=yy,
                mode="lines",
                name="Densité normale",
                line=dict(
                    color=couleurs["primaire"],
                    width=3
                )
            ))

            masque = xx <= cible

            fig.add_trace(go.Scatter(
                x=xx[masque],
                y=yy[masque],
                mode="lines",
                fill="tozeroy",
                name="Probabilité sous la cible",
                line=dict(width=0),
                fillcolor="rgba(34,166,153,0.35)"
            ))

            fig.add_vline(
                x=cible,
                line_dash="dash",
                line_color=couleurs["danger"],
                annotation_text="Résistance cible"
            )

            fig.update_layout(
                xaxis_title="Résistance (MPa)",
                yaxis_title="Densité de probabilité"
            )

            st.plotly_chart(
                appliquer_style_graphique(
                    fig,
                    "Distribution normale estimée"
                ),
                use_container_width=True
            )

            st.warning(
                "Ces probabilités sont des estimations fondées sur "
                "un modèle normal. Elles ne garantissent pas la "
                "conformité réglementaire du béton."
            )


# =========================================================
# 14. DIAGRAMMES
# =========================================================

elif page == "Diagrammes":

    st.header("📈 Galerie de diagrammes")

    if df.empty:
        st.info(
            "Enregistrez des essais pour générer les diagrammes."
        )

    else:
        choix = st.selectbox(
            "Choisir le diagramme à afficher",
            [
                "Histogramme des résistances",
                "Évolution dans le temps",
                "Boîte à moustaches",
                "Distribution normale",
                "Résistance en fonction de l'âge"
            ]
        )

        if choix == "Histogramme des résistances":
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
                color="sample_code",
                hover_data=["test_date"],
                labels={
                    "age_days": "Âge du béton (jours)",
                    "strength_mpa": "Résistance (MPa)",
                    "sample_code": "Éprouvette"
                }
            )

            fig = appliquer_style_graphique(
                fig,
                "Résistance en fonction de l'âge du béton"
            )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if st.download_button(
            "⬇️ Exporter les données des diagrammes",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="betonlab_donnees_graphiques.csv",
            mime="text/csv"
        ):
            enregistrer_action(
                "Exportation des données des diagrammes",
                f"{len(df)} essai(s)"
            )


# =========================================================
# 15. HISTORIQUE D'UTILISATION
# =========================================================

elif page == "Historique d'utilisation":

    st.header("🕒 Historique d'utilisation")

    st.write(
        "Consultez les actions enregistrées par l'application : "
        "ajouts, suppressions, consultations et exportations."
    )

    historique = charger_historique()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Actions enregistrées",
            len(historique)
        )

    with c2:
        st.metric(
            "Essais dans la base",
            len(df)
        )

    with c3:
        st.metric(
            "Rubrique actuelle",
            "Historique"
        )

    if historique.empty:
        st.info("Aucune action n'a encore été enregistrée.")

    else:
        actions = sorted(
            historique["Action"].dropna().unique().tolist()
        )

        filtre = st.selectbox(
            "Filtrer les actions",
            ["Toutes les actions"] + actions
        )

        historique_vue = historique.copy()

        if filtre != "Toutes les actions":
            historique_vue = historique_vue[
                historique_vue["Action"] == filtre
            ]

        st.dataframe(
            historique_vue,
            use_container_width=True,
            hide_index=True
        )

        if st.download_button(
            "⬇️ Exporter l'historique en CSV",
            data=historique_vue.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="betonlab_historique.csv",
            mime="text/csv"
        ):
            enregistrer_action(
                "Exportation de l'historique",
                f"{len(historique_vue)} action(s) exportée(s)"
            )


# =========================================================
# 16. PIED DE PAGE
# =========================================================

st.divider()

st.caption(
    f"BetonLab • Analyse du béton • "
    f"Thème : {st.session_state.theme_betonlab} • "
    "Résistance exprimée en MPa"
)
