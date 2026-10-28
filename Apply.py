import sqlite3
from pathlib import Path
from datetime import date
from statistics import NormalDist

import pandas as pd
import streamlit as st


# BETONLAB_CUSTOM_STYLE
st.markdown("""
<style>
/* Overall page */
.stApp {
    background: linear-gradient(135deg, #f3f7fb 0%, #e8f0f7 100%);
    color: #17324d;
}
/* Main title and headings */
h1, h2, h3 {
    color: #123b5d !important;
}
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123b5d 0%, #176b87 100%);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: #087e8b;
    color: #ffffff;
    border: 0;
    border-radius: 9px;
    padding: 0.55rem 1rem;
    font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #065f68;
    color: #ffffff;
    border: 0;
}
/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d6e3ed;
    border-left: 5px solid #087e8b;
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 2px 8px rgba(18, 59, 93, 0.06);
}
/* Inputs and content containers */
[data-testid="stForm"], [data-testid="stExpander"] {
    border-radius: 10px;
}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    border-radius: 7px;
}
/* Tabs */
button[data-baseweb="tab"] {
    color: #123b5d;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom-color: #087e8b;
    color: #087e8b;
}
/* Alerts */
[data-testid="stAlert"] {
    border-radius: 9px;
}
</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "betonlab.db"

st.set_page_config(
    page_title="BetonLab",
    page_icon="🏗️",
    layout="wide"
)


# --------------------------------------------------
# 2. SQLITE DATABASE
# --------------------------------------------------

def connect_database():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with connect_database() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                specimen_reference TEXT NOT NULL UNIQUE,
                casting_date TEXT,
                test_date TEXT NOT NULL,
                age_days INTEGER NOT NULL CHECK(age_days >= 0),
                specimen_shape TEXT NOT NULL,
                dimension_mm REAL NOT NULL CHECK(dimension_mm > 0),
                failure_load_kn REAL NOT NULL CHECK(failure_load_kn > 0),
                strength_mpa REAL NOT NULL
            )
        """)


def add_test(data):
    with connect_database() as db:
        db.execute("""
            INSERT INTO tests (
                project, specimen_reference, casting_date, test_date,
                age_days, specimen_shape, dimension_mm, failure_load_kn,
                strength_mpa
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)


def read_tests():
    with connect_database() as db:
        rows = db.execute("""
            SELECT * FROM tests
            ORDER BY test_date DESC, id DESC
        """).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def delete_test(test_id):
    with connect_database() as db:
        db.execute("DELETE FROM tests WHERE id = ?", (test_id,))


initialize_database()


# --------------------------------------------------
# 3. CONCRETE COMPRESSIVE STRENGTH
# --------------------------------------------------

def calculate_strength(failure_load_kn, shape, dimension_mm):
    """
    Calculate compressive strength in MPa.
    Cube: dimension_mm is the side length.
    Cylinder: dimension_mm is the diameter.
    1 N/mm² = 1 MPa.
    """
    if shape == "Cube":
        area_mm2 = dimension_mm ** 2
    else:
        area_mm2 = 3.141592653589793 * dimension_mm ** 2 / 4

    return failure_load_kn * 1000 / area_mm2


# --------------------------------------------------
# 4. APPLICATION HEADER AND NAVIGATION
# --------------------------------------------------

st.title("🏗️ BetonLab")
st.caption(
    "Concrete compression test collection, statistical analysis, "
    "probability estimates, and charts"
)

entry_tab, analysis_tab, probability_tab, history_tab = st.tabs([
    "📝 Test Entry",
    "📊 Analysis",
    "🎲 Probability & Charts",
    "📚 Test History"
])


# --------------------------------------------------
# 5. TEST ENTRY FORM
# --------------------------------------------------

with entry_tab:
    st.subheader("Record a compression test")

    with st.form("test_entry_form"):
        project = st.text_input("Project / construction site")
        specimen_reference = st.text_input("Unique specimen reference")

        col1, col2 = st.columns(2)

        with col1:
            casting_date = st.date_input("Casting date", value=date.today())
            test_date = st.date_input("Test date", value=date.today())
            specimen_shape = st.selectbox("Specimen shape", ["Cube", "Cylinder"])

        with col2:
            dimension = st.number_input(
                "Cube: side length / cylinder: diameter (mm)",
                min_value=1.0,
                value=150.0,
                step=5.0
            )
            failure_load = st.number_input(
                "Maximum failure load (kN)",
                min_value=0.1,
                value=450.0,
                step=10.0
            )

        age_entered = st.number_input(
            "Concrete age (days)",
            min_value=0,
            value=7,
            step=1
        )

        submitted = st.form_submit_button("Save test", type="primary")

    strength = calculate_strength(failure_load, specimen_shape, dimension)
    st.metric("Calculated compressive strength", f"{strength:.2f} MPa")

    if submitted:
        project = project.strip()
        specimen_reference = specimen_reference.strip()

        if not project or not specimen_reference:
            st.error("Project and specimen reference are required.")
        elif test_date < casting_date:
            st.error("The test date cannot be earlier than the casting date.")
        else:
            actual_age = (test_date - casting_date).days

            if int(age_entered) != actual_age:
                st.error(
                    f"The entered age ({age_entered} days) does not match "
                    f"the dates ({actual_age} days). Please check the values."
                )
            else:
                try:
                    add_test((
                        project,
                        specimen_reference,
                        casting_date.isoformat(),
                        test_date.isoformat(),
                        int(age_entered),
                        specimen_shape,
                        float(dimension),
                        float(failure_load),
                        float(strength)
                    ))
                    st.success("Test saved to the local SQLite database.")
                except sqlite3.IntegrityError:
                    st.error("This specimen reference already exists.")


# --------------------------------------------------
# 6. LOAD DATA
# --------------------------------------------------

df = read_tests()


# --------------------------------------------------
# 7. GENERAL ANALYSIS
# --------------------------------------------------

with analysis_tab:
    st.subheader("Dashboard")

    if df.empty:
        st.info("No tests have been recorded yet. Add a test in the Test Entry tab.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Number of tests", len(df))
        col2.metric("Mean strength", f'{df["strength_mpa"].mean():.2f} MPa')
        col3.metric("Minimum strength", f'{df["strength_mpa"].min():.2f} MPa')
        col4.metric("Maximum strength", f'{df["strength_mpa"].max():.2f} MPa')

        st.subheader("Mean strength by concrete age")

        age_summary = (
            df.groupby("age_days")["strength_mpa"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("age_days")
        )
        age_summary.columns = [
            "Age (days)", "Mean strength (MPa)", "Number of tests"
        ]
        st.dataframe(age_summary, hide_index=True, use_container_width=True)

        line_data = (
            df.groupby("age_days")["strength_mpa"]
            .mean()
            .sort_index()
            .rename("Mean strength (MPa)")
        )
        st.line_chart(line_data)

        st.caption(
            "This chart shows the mean recorded strength for each available age. "
            "Interpret results according to the applicable test standard and project criteria."
        )


# --------------------------------------------------
# 8. PROBABILITY ANALYSIS AND CHARTS
# --------------------------------------------------

with probability_tab:
    st.subheader("Probability analysis")

    if df.empty:
        st.info("Record at least one test to calculate empirical probabilities and display charts.")
    else:
        strengths = pd.to_numeric(df["strength_mpa"], errors="coerce").dropna()

        st.markdown(
            "Enter a target compressive strength. BetonLab will calculate the "
            "observed proportion of recorded tests that meet the target."
        )
        target_strength = st.number_input(
            "Target compressive strength (MPa)",
            min_value=0.1,
            value=25.0,
            step=1.0,
            key="target_strength"
        )

        total_tests = len(strengths)
        meeting_target = int((strengths >= target_strength).sum())
        below_target = int((strengths < target_strength).sum())
        empirical_success = meeting_target / total_tests
        empirical_below = below_target / total_tests

        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Observed probability of meeting target",
            f"{empirical_success * 100:.1f}%"
        )
        p2.metric(
            "Observed probability below target",
            f"{empirical_below * 100:.1f}%"
        )
        p3.metric(
            "Tests meeting target",
            f"{meeting_target} / {total_tests}"
        )

        probability_table = pd.DataFrame({
            "Outcome": ["Meets or exceeds target", "Below target"],
            "Number of tests": [meeting_target, below_target],
            "Empirical probability (%)": [
                empirical_success * 100,
                empirical_below * 100
            ]
        })
        st.subheader("Observed outcome proportions")
        st.dataframe(probability_table, hide_index=True, use_container_width=True)
        st.bar_chart(probability_table.set_index("Outcome")["Empirical probability (%)"])

        st.subheader("Distribution of recorded strengths")
        number_of_bins = min(10, max(1, int(len(strengths) ** 0.5)))
        if strengths.nunique() == 1:
            st.info("All recorded strengths are identical, so a histogram cannot show a spread.")
        else:
            histogram_counts = pd.cut(
                strengths,
                bins=number_of_bins,
                include_lowest=True
            ).value_counts(sort=False)
            histogram_counts.index = histogram_counts.index.astype(str)
            histogram_df = histogram_counts.rename("Number of tests").to_frame()
            st.bar_chart(histogram_df)
            st.caption("Each bar represents the number of recorded tests within a strength interval.")

        st.subheader("Strength versus specimen age")
        scatter_data = df[["age_days", "strength_mpa"]].dropna().rename(
            columns={
                "age_days": "Age (days)",
                "strength_mpa": "Strength (MPa)"
            }
        )
        st.scatter_chart(
            scatter_data,
            x="Age (days)",
            y="Strength (MPa)"
        )

        st.subheader("Optional normal-distribution estimate")
        st.warning(
            "This is a theoretical approximation, not a certified design probability. "
            "It assumes the recorded strengths follow a normal distribution. "
            "Small or unrepresentative samples can produce misleading estimates."
        )

        mean_strength = float(strengths.mean())
        std_strength = float(strengths.std(ddof=1)) if len(strengths) > 1 else 0.0

        if len(strengths) < 2 or std_strength <= 0:
            st.info(
                "At least two tests with some variation in strength are required "
                "for this normal-distribution estimate."
            )
        else:
            normal_probability = 1 - NormalDist(
                mu=mean_strength,
                sigma=std_strength
            ).cdf(target_strength)
            normal_probability = min(1.0, max(0.0, normal_probability))
            st.metric(
                "Estimated normal-model probability of meeting target",
                f"{normal_probability * 100:.1f}%"
            )
            st.caption(
                f"Model parameters: mean = {mean_strength:.2f} MPa; "
                f"sample standard deviation = {std_strength:.2f} MPa."
            )

        st.caption(
            "Empirical probabilities describe only the tests entered into this application. "
            "They should not be treated as proof that an entire concrete batch or structure "
            "complies with a standard."
        )


# --------------------------------------------------
# 9. TEST HISTORY, EXPORT, AND DELETION
# --------------------------------------------------

with history_tab:
    st.subheader("Test history")

    if df.empty:
        st.info("The database is empty.")
    else:
        column_names = {
            "id": "ID",
            "project": "Project",
            "specimen_reference": "Specimen reference",
            "casting_date": "Casting date",
            "test_date": "Test date",
            "age_days": "Age (days)",
            "specimen_shape": "Shape",
            "dimension_mm": "Dimension (mm)",
            "failure_load_kn": "Failure load (kN)",
            "strength_mpa": "Strength (MPa)"
        }
        table = df.rename(columns=column_names)

        st.dataframe(
            table.drop(columns=["ID"]),
            hide_index=True,
            use_container_width=True
        )

        csv_data = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Export results as CSV",
            data=csv_data,
            file_name="betonlab_test_results.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("Delete a test")

        test_ids = df["id"].tolist()
        selected_id = st.selectbox("Select the test ID", test_ids)
        confirm_delete = st.checkbox("I confirm that I want to delete this test")

        if st.button("Delete test"):
            if confirm_delete:
                delete_test(int(selected_id))
                st.success("Test deleted.")
                st.rerun()
            else:
                st.warning("Check the confirmation box before deleting.")

        st.caption(
            f"Local database: {DB_PATH.name}. Make regular backup copies of this file."
        )
