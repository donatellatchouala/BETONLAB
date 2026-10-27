
import sqlite3
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DOSSIER = Path(__file__).resolve().parent
DB_PATH = DOSSIER / "betonlab.db"

st.set_page_config(page_title="BetonLab", page_icon="🏗️", layout="wide")


# --------------------------------------------------
# 2. BASE DE DONNEES SQLITE
# --------------------------------------------------

def connecter():
    connexion = sqlite3.connect(DB_PATH)
    connexion.row_factory = sqlite3.Row
    return connexion


def initialiser_base():
    with connecter() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS essais (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chantier TEXT NOT NULL,
                reference TEXT NOT NULL UNIQUE,
                date_coulage TEXT,
                date_essai TEXT NOT NULL,
                age_jours INTEGER NOT NULL CHECK(age_jours >= 0),
                forme TEXT NOT NULL,
                dimension_mm REAL NOT NULL CHECK(dimension_mm > 0),
                charge_kn REAL NOT NULL CHECK(charge_kn > 0),
                resistance_mpa REAL NOT NULL
            )
        """)


def ajouter_essai(donnees):
    with connecter() as db:
        db.execute("""
            INSERT INTO essais (
                chantier, reference, date_coulage, date_essai,
                age_jours, forme, dimension_mm, charge_kn,
                resistance_mpa
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, donnees)


def lire_essais():
    with connecter() as db:
        lignes = db.execute("""
            SELECT * FROM essais
            ORDER BY date_essai DESC, id DESC
        """).fetchall()

    return pd.DataFrame([dict(ligne) for ligne in lignes])


def supprimer_essai(identifiant):
    with connecter() as db:
        db.execute(
            "DELETE FROM essais WHERE id = ?",
            (identifiant,)
        )


initialiser_base()


# --------------------------------------------------
# 3. CALCUL DE RESISTANCE
# --------------------------------------------------

def calculer_resistance(charge_kn, forme, dimension_mm):
    """
    Hypothèse : section carrée ou circulaire,
    charge en kN, dimensions en mm.
    1 N/mm² = 1 MPa.
    """
    if forme == "Cube":
        surface = dimension_mm ** 2
    else:
        surface = 3.141592653589793 * dimension_mm ** 2 / 4

    return charge_kn * 1000 / surface


# --------------------------------------------------
# 4. INTERFACE
# --------------------------------------------------

st.title("🏗️ BetonLab")
st.caption(
    "Collecte et analyse des résultats "
    "d'essais de compression du béton"
)

saisie, analyse, historique = st.tabs([
    "📝 Saisie des essais",
    "📊 Analyse",
    "📚 Historique"
])


# --------------------------------------------------
# 5. FORMULAIRE DE SAISIE
# --------------------------------------------------

with saisie:
    st.subheader("Enregistrer un essai")

    with st.form("formulaire_essai"):
        chantier = st.text_input(
           "Nom du chantier / projet"
        )
        reference = st.text_input(
            "Référence unique de l'éprouvette"
        )

        col1, col2 = st.columns(2)

        with col1:
            date_coulage = st.date_input(
               "Date de coulage",
                value=date.today()
            )
            date_essai = st.date_input(
                "Date de l'essai",
                value=date.today()
            )
            forme = st.selectbox(
                "Forme de l'éprouvette",
                ["Cube", "Cylindre"]
            )

        with col2:
            dimension = st.number_input(
                "Cube : arête / cylindre : diamètre (mm)",
                min_value=1.0,
                value=150.0,
                step=5.0
            )
            charge = st.number_input(
                "Charge maximale de rupture (kN)",
                min_value=0.1,
                value=450.0,
                step=10.0
            )

        age_saisi = st.number_input(
            "Âge du béton (jours)",
            min_value=0,
            value=7,
            step=1
        )

        valider = st.form_submit_button(
            "Enregistrer l'essai",
           type="primary"
        )

    resistance = calculer_resistance(
        charge, forme, dimension
    )

    st.metric(
        "Résistance calculée",
        f"{resistance:.2f} MPa"
    )

    if valider:
        reference = reference.strip()
        chantier = chantier.strip()

        if not chantier or not reference:
            st.error(
                "Le chantier et la référence sont obligatoires."
            )

        elif date_essai < date_coulage:
            st.error(
                "La date de l'essai précède la date de coulage."
            )

        else:
            age_reel = (date_essai - date_coulage).days

            if int(age_saisi) != age_reel:
                st.error(
                    f"L'âge indiqué ({age_saisi} jours) "
                    f"ne correspond pas aux dates "
                    f"({age_reel} jours). Vérifie les données."
                )
            else:
                try:
                    ajouter_essai((
                        chantier,
                        reference,
                        date_coulage.isoformat(),
                        date_essai.isoformat(),
                        int(age_saisi),
                        forme,
                        float(dimension),
                        float(charge),
                        float(resistance)
                    ))
                    st.success(
                        "Essai enregistré durablement dans SQLite."
                    )
                except sqlite3.IntegrityError:
                    st.error(
                        "Cette référence existe déjà."
                    )


# --------------------------------------------------
# 6. CHARGEMENT DES DONNEES
# --------------------------------------------------

df = lire_essais()


# --------------------------------------------------
# 7. ANALYSE ET GRAPHIQUES
# --------------------------------------------------

with analyse:
    st.subheader("Tableau de bord")

    if df.empty:
        st.info(
            "Aucun essai enregistré. "
            "Ajoute un essai dans l'onglet Saisie."
        )
    else:
        c1, c2, c3 = st.columns(3)

        c1.metric("Nombre d'essais", len(df))

        c2.metric(
            "Résistance moyenne",
            f'{df["resistance_mpa"].mean():.2f} MPa'
        )

        c3.metric(
            "Résistance minimale",
            f'{df["resistance_mpa"].min():.2f} MPa'
        )

        st.subheader("Résistance selon l'âge")

        resume = (
            df.groupby("age_jours")["resistance_mpa"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("age_jours")
        )

        resume.columns = [
            "Âge (jours)",
            "Résistance moyenne (MPa)",
            "Nombre d'essais"
        ]

        st.dataframe(
            resume,
            hide_index=True,
            use_container_width=True
        )

        graphique = (
            df.groupby("age_jours")["resistance_mpa"]
            .mean()
            .sort_index()
        )

        st.line_chart(graphique)

        st.caption(
            "La courbe présente la moyenne des essais "
            "pour chaque âge disponible. Les résultats "
            "doivent être interprétés selon le protocole "
            "et les critères applicables au projet."
        )


# --------------------------------------------------
# 8. HISTORIQUE, EXPORT ET SUPPRESSION
# --------------------------------------------------

with historique:
    st.subheader("Historique des essais")

    if df.empty:
        st.info("La base de données est vide.")
    else:
        colonnes = {
            "id": "ID",
            "chantier": "Chantier",
            "reference": "Référence",
            "date_coulage": "Date de coulage",
            "date_essai": "Date de l'essai",
            "age_jours": "Âge (jours)",
            "forme": "Forme",
            "dimension_mm": "Dimension (mm)",
            "charge_kn": "Charge (kN)",
            "resistance_mpa": "Résistance (MPa)"
        }

        tableau = df.rename(columns=colonnes)

        st.dataframe(
            tableau.drop(columns=["ID"]),
            hide_index=True,
            use_container_width=True
        )

        csv = tableau.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Exporter les résultats en CSV",
            data=csv,
            file_name="resultats_betonlab.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("Supprimer un essai")

        identifiants = df["id"].tolist()

        choix = st.selectbox(
            "Sélectionner l'identifiant de l'essai",
            identifiants
        )

        confirmer = st.checkbox(
            "Je confirme la suppression de cet essai"
        )

        if st.button("Supprimer l'essai"):
            if confirmer:
                supprimer_essai(int(choix))
                st.success("Essai supprimé.")
                st.rerun()
            else:
                st.warning(
                    "Coche la confirmation avant de supprimer."
                )

        st.caption(
            f"Base locale : {DB_PATH.name}. "
            "Sauvegarde régulièrement ce fichier."
        )

