import os
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from PIL import Image

# =========================================================
# 1. CONFIGURATION GENERALE
# =========================================================

st.set_page_config(
    page_title="BetonLab | Concrete Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "betonlab.db")

# =========================================================
# 2. CONFIGURATION DES COULEURS
# Modifie ces valeurs pour personnaliser l'application.
# =========================================================

COLORS = {
    "primary": "#173B67",
    "secondary": "#246B8E",
    "accent": "#22A699",
    "background": "#F3F6FB",
    "panel": "#FFFFFF",
    "text": "#172B4D",
    "muted": "#64748B",
    "danger": "#D64545"
}

# =========================================================
# 3. STYLE CSS
# =========================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLORS["background"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {COLORS["primary"]};
    }}

    [data-testid="stSidebar"] * {{
        color: white;
    }}

    h1, h2, h3 {{
        color: {COLORS["primary"]};
    }}

    div.stButton > button {{
        background-color: {COLORS["accent"]};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }}

    div.stButton > button:hover {{
        background-color: {COLORS["secondary"]};
        color: white;
    }}

    [data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #DCE4EF;
        padding: 16px;
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 4. BASE DE DONNEES SQLITE
# =========================================================

def initialize_database():
    with sqlite3.connect(DB_PATH, timeout=10) as con:
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

        con.commit()


@st.cache_data(ttl=30, show_spinner=False)
def load_data():
    """Charge les données et met le résultat en cache."""
    with sqlite3.connect(DB_PATH, timeout=10) as con:
        df = pd.read_sql_query(
            "SELECT * FROM tests ORDER BY test_date DESC, id DESC",
            con
        )

    return df


def save_test(
    code,
    test_date,
    age,
    length,
    width,
    height,
    load,
    strength
):
    with sqlite3.connect(DB_PATH, timeout=10) as con:
        con.execute("""
            INSERT INTO tests (
                sample_code, test_date, age_days,
                length_mm, width_mm, height_mm,
                load_kn, strength_mpa
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            code,
            test_date,
            age,
            length,
            width,
            height,
            load,
            strength
        ))

        con.commit()

    load_data.clear()


# =========================================================
# 5. CALCUL DE LA RESISTANCE
# =========================================================

def calculate_strength(load_kn, length_mm, width_mm):
    area = length_mm * width_mm

    if area <= 0:
        raise ValueError("The loaded area must be positive.")

    # 1 kN = 1000 N
    # 1 MPa = 1 N/mm²
    return load_kn * 1000 / area


# =========================================================
# 6. ANALYSE STATISTIQUE
# =========================================================

def calculate_statistics(df):
    x = df["strength_mpa"].to_numpy(dtype=float)

    if len(x) == 0:
        return {}

    result = {
        "count": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "range": float(np.ptp(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else None
    }

    if result["std"] is not None and result["mean"] != 0:
        result["cv"] = result["std"] / abs(result["mean"]) * 100
    else:
        result["cv"] = None

    return result


# =========================================================
# 7. GRAPHIQUES PLOTLY
# =========================================================

def apply_chart_style(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(
            family="Arial",
            size=13,
            color=COLORS["text"]
        ),
        title_font=dict(
            size=20,
            color=COLORS["primary"]
        ),
        margin=dict(l=30, r=30, t=70, b=35),
        hovermode="closest"
    )

    return fig


def histogram_chart(df):
    fig = px.histogram(
        df,
        x="strength_mpa",
        nbins="auto",
        color_discrete_sequence=[COLORS["secondary"]],
        labels={
            "strength_mpa": "Compressive strength (MPa)",
            "count": "Number of tests"
        }
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=1
    )

    return apply_chart_style(
        fig,
        "Distribution of compressive strength"
    )


def time_chart(df):
    ordered = df.sort_values("test_date")

    fig = px.line(
        ordered,
        x="test_date",
        y="strength_mpa",
        markers=True,
        color_discrete_sequence=[COLORS["accent"]],
        labels={
            "test_date": "Test date",
            "strength_mpa": "Strength (MPa)"
        }
    )

    return apply_chart_style(
        fig,
        "Compressive strength over time"
    )


def box_chart(df):
    fig = px.box(
        df,
        y="strength_mpa",
        points="all",
        color_discrete_sequence=[COLORS["accent"]],
        labels={
            "strength_mpa": "Strength (MPa)"
        }
    )

    return apply_chart_style(
        fig,
        "Strength variability"
    )


def normal_chart(df):
    x = df["strength_mpa"].to_numpy(dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=x,
            histnorm="probability density",
            name="Observed data",
            marker_color=COLORS["secondary"],
            opacity=0.55,
            nbinsx="auto"
        )
    )

    if len(x) >= 2:
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))

        if sd > 0:
            xx = np.linspace(
                float(np.min(x)),
                float(np.max(x)),
                250
            )

            yy = stats.norm.pdf(xx, mean, sd)

            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=yy,
                    mode="lines",
                    name="Normal model",
                    line=dict(
                        color=COLORS["accent"],
                        width=3
                    )
                )
            )

    fig.update_layout(
        xaxis_title="Strength (MPa)",
        yaxis_title="Probability density"
    )

    return apply_chart_style(
        fig,
        "Observed distribution and normal model"
    )


# =========================================================
# 8. EN-TETE ET IMAGE VECTORIELLE
# =========================================================

st.markdown(
    f"""
    <div style="
        background: linear-gradient(
            110deg,
            {COLORS['primary']},
            {COLORS['secondary']}
        );
        padding: 26px;
        border-radius: 16px;
        margin-bottom: 20px;
    ">
        <h1 style="color:white;margin:0;">
            🏗️ BetonLab
        </h1>
        <p style="color:white;font-size:16px;margin-bottom:0;">
            Concrete Compression Test Analyzer
        </p>
        <p style="color:#D8F3EF;margin-top:8px;">
            Engineering • Statistics • Probability • Data Visualization
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 9. MENU LATERAL
# =========================================================

with st.sidebar:
    st.title("BETONLAB")
    st.caption("Concrete laboratory dashboard")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Add a test",
            "Database",
            "Statistics",
            "Probability",
            "Charts"
        ]
    )

    st.divider()

    st.caption("Version 1.0")
    st.caption("Python + Streamlit + Plotly")


# =========================================================
# 10. CHARGEMENT DES DONNEES
# =========================================================

initialize_database()
df = load_data()


# =========================================================
# 11. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header("Dashboard")

    if df.empty:
        st.info(
            "No tests recorded yet. "
            "Open 'Add a test' to enter your first result."
        )

    else:
        s = calculate_statistics(df)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Number of tests",
            s["count"]
        )

        c2.metric(
            "Mean strength",
            f'{s["mean"]:.2f} MPa'
        )

        c3.metric(
            "Maximum strength",
            f'{s["maximum"]:.2f} MPa'
        )

        c4.metric(
            "Minimum strength",
            f'{s["minimum"]:.2f} MPa'
        )

        st.subheader("Recent tests")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                histogram_chart(df),
                use_container_width=True
            )

        with right:
            st.plotly_chart(
                time_chart(df),
                use_container_width=True
            )


# =========================================================
# 12. AJOUT D'UN ESSAI
# =========================================================

elif page == "Add a test":

    st.header("Record a compression test")

    with st.form("test_form", clear_on_submit=True):

        c1, c2 = st.columns(2)

        with c1:
            code = st.text_input(
                "Sample code",
                placeholder="B-001"
            )

            test_date = st.date_input(
                "Test date",
                value=date.today()
            )

            age = st.number_input(
                "Concrete age (days)",
                min_value=1,
                max_value=36500,
                value=28
            )

            length = st.number_input(
                "Length (mm)",
                min_value=0.1,
                value=150.0
            )

        with c2:
            width = st.number_input(
                "Width (mm)",
                min_value=0.1,
                value=150.0
            )

            height = st.number_input(
                "Height (mm)",
                min_value=0.1,
                value=300.0
            )

            load = st.number_input(
                "Maximum load (kN)",
                min_value=0.1,
                value=650.0
            )

        submitted = st.form_submit_button(
            "Save compression test",
            use_container_width=True
        )

    if submitted:

        if not code.strip():
            st.error("Please enter a sample code.")

        else:
            strength = calculate_strength(
                load,
                length,
                width
            )

            save_test(
                code.strip(),
                test_date.isoformat(),
                int(age),
                float(length),
                float(width),
                float(height),
                float(load),
                float(strength)
            )

            st.success("Test saved successfully.")

            st.metric(
                "Calculated compressive strength",
                f"{strength:.2f} MPa"
            )

            st.info(
                "Calculation: load (kN) × 1000 / "
                "loaded area (mm²)."
            )


# =========================================================
# 13. BASE DE DONNEES
# =========================================================

elif page == "Database":

    st.header("Test database")

    if df.empty:
        st.info("The database is empty.")

    else:
        search = st.text_input(
            "Search by sample code"
        )

        view = df.copy()

        if search.strip():
            view = view[
                view["sample_code"].str.contains(
                    search.strip(),
                    case=False,
                    na=False
                )
            ]

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True
        )

        csv_data = view.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Download results as CSV",
            data=csv_data,
            file_name="betonlab_results.csv",
            mime="text/csv"
        )

        st.warning(
            "Deleting a record permanently removes it from "
            "the current database."
        )

        selected_id = st.number_input(
            "ID of the test to delete",
            min_value=1,
            step=1
        )

        if st.button("Delete selected test"):

            with sqlite3.connect(DB_PATH) as con:
                cursor = con.execute(
                    "DELETE FROM tests WHERE id = ?",
                    (int(selected_id),)
                )
                con.commit()

            if cursor.rowcount:
                load_data.clear()
                st.success("Test deleted.")
                st.rerun()

            else:
                st.error("No test found with this ID.")


# =========================================================
# 14. STATISTIQUES
# =========================================================

elif page == "Statistics":

    st.header("Statistical analysis")

    if df.empty:
        st.info("Record some tests first.")

    else:
        s = calculate_statistics(df)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean",
            f'{s["mean"]:.2f} MPa'
        )

        c2.metric(
            "Median",
            f'{s["median"]:.2f} MPa'
        )

        c3.metric(
            "Range",
            f'{s["range"]:.2f} MPa'
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Standard deviation",
            f'{s["std"]:.2f} MPa'
            if s["std"] is not None else "N/A"
        )

        c2.metric(
            "Coefficient of variation",
            f'{s["cv"]:.2f}%'
            if s["cv"] is not None else "N/A"
        )

        c3.metric(
            "Number of tests",
            s["count"]
        )

        st.subheader("Complete statistical table")

        summary = pd.DataFrame({
            "Statistic": [
                "Count",
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "Range",
                "Standard deviation",
                "Coefficient of variation"
            ],
            "Value": [
                s["count"],
                s["mean"],
                s["median"],
                s["minimum"],
                s["maximum"],
                s["range"],
                s["std"],
                s["cv"]
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.plotly_chart(
            box_chart(df),
            use_container_width=True
        )

        if len(df) >= 3:
            x = df["strength_mpa"].to_numpy(dtype=float)

            if np.std(x) > 0:
                statistic, p_value = stats.shapiro(x)

                st.subheader("Normality diagnostic")
                st.write(
                    f"Shapiro-Wilk statistic: {statistic:.4f}"
                )
                st.write(
                    f"P-value: {p_value:.4f}"
                )

                st.caption(
                    "A p-value below 0.05 is evidence against "
                    "normality under this test. Results depend "
                    "on sample size; this is not proof that a "
                    "distribution is normal."
                )


# =========================================================
# 15. PROBABILITES
# =========================================================

elif page == "Probability":

    st.header("Probability analysis")

    st.write(
        "Estimate probabilities under a normal distribution "
        "fitted to the recorded strengths."
    )

    if len(df) < 2:
        st.warning(
            "At least two recorded tests are required."
        )

    else:
        x = df["strength_mpa"].to_numpy(dtype=float)

        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))

        st.metric(
            "Mean strength",
            f"{mean:.2f} MPa"
        )

        st.metric(
            "Sample standard deviation",
            f"{sd:.2f} MPa"
        )

        target = st.number_input(
            "Target strength (MPa)",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        if sd == 0:
            st.warning(
                "All measured strengths are identical. "
                "A fitted normal distribution cannot be "
                "estimated with zero standard deviation."
            )

        else:
            p_below = float(
                stats.norm.cdf(
                    target,
                    loc=mean,
                    scale=sd
                )
            )

            p_above = 1.0 - p_below

            c1, c2 = st.columns(2)

            c1.metric(
                f"P(X ≤ {target:.2f})",
                f"{100 * p_below:.2f}%"
            )

            c2.metric(
                f"P(X ≥ {target:.2f})",
                f"{100 * p_above:.2f}%"
            )

            xx = np.linspace(
                min(float(np.min(x)), mean - 4 * sd),
                max(float(np.max(x)), mean + 4 * sd),
                400
            )

            yy = stats.norm.pdf(
                xx,
                loc=mean,
                scale=sd
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=xx,
                    y=yy,
                    mode="lines",
                    name="Normal density",
                    line=dict(
                        color=COLORS["secondary"],
                        width=3
                    )
                )
            )

            mask = xx <= target

            fig.add_trace(
                go.Scatter(
                    x=xx[mask],
                    y=yy[mask],
                    fill="tozeroy",
                    mode="none",
                    name="Below target",
                    fillcolor="rgba(34,166,153,0.35)"
                )
            )

            fig.add_vline(
                x=target,
                line_dash="dash",
                line_color=COLORS["danger"],
                annotation_text="Target"
            )

            fig.update_layout(
                xaxis_title="Strength (MPa)",
                yaxis_title="Probability density"
            )

            st.plotly_chart(
                apply_chart_style(
                    fig,
                    "Normal-model probability"
                ),
                use_container_width=True
            )

            st.caption(
                "These are model-based estimates, not a guarantee "
                "of concrete conformity. Interpret them using "
                "the applicable testing standard and sufficient "
                "representative observations."
            )


# =========================================================
# 16. GRAPHIQUES
# =========================================================

elif page == "Charts":

    st.header("Interactive charts")

    if df.empty:
        st.info("Record tests to create charts.")

    else:
        chart_choice = st.selectbox(
            "Choose a chart",
            [
                "Strength histogram",
                "Strength over time",
                "Box plot",
                "Normal distribution"
            ]
        )

        if chart_choice == "Strength histogram":
            fig = histogram_chart(df)

        elif chart_choice == "Strength over time":
            fig = time_chart(df)

        elif chart_choice == "Box plot":
            fig = box_chart(df)

        else:
            fig = normal_chart(df)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.download_button(
            "Download chart data as CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="betonlab_chart_data.csv",
            mime="text/csv"
        )


# =========================================================
# 17. PIED DE PAGE
# =========================================================

st.divider()

st.caption(
    "BetonLab | Concrete Compression Test Analyzer | "
    "Statistical estimates should be interpreted with "
    "the applicable engineering standards."
)
