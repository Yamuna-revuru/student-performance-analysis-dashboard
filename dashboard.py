
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# ---------------- LOAD DATA ----------------
data_path = Path(__file__).parent / "student-por.csv"

st.title("🎓 Student Performance Analysis Dashboard")
st.write(
    "Analyze student grades, study habits, past failures, "
    "and attendance patterns using interactive visualizations."
)

try:
    df = pd.read_csv(data_path, sep=";")
except FileNotFoundError:
    st.error("Dataset not found. Keep student-por.csv beside dashboard.py.")
    st.stop()

required_columns = [
    "school", "sex", "studytime", "failures",
    "absences", "G1", "G2", "G3"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Required columns are missing: {missing_columns}")
    st.stop()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Dashboard Filters")

schools = st.sidebar.multiselect(
    "Select School",
    options=sorted(df["school"].dropna().unique().tolist()),
    default=sorted(df["school"].dropna().unique().tolist())
)

genders = st.sidebar.multiselect(
    "Select Gender",
    options=sorted(df["sex"].dropna().unique().tolist()),
    default=sorted(df["sex"].dropna().unique().tolist())
)

study_times = st.sidebar.multiselect(
    "Select Study Time",
    options=sorted(df["studytime"].dropna().unique().tolist()),
    default=sorted(df["studytime"].dropna().unique().tolist())
)

min_grade, max_grade = st.sidebar.slider(
    "Final Grade Range",
    min_value=0,
    max_value=20,
    value=(0, 20)
)

# ---------------- FILTER DATA ----------------
df = df[
    df["school"].isin(schools)
    & df["sex"].isin(genders)
    & df["studytime"].isin(study_times)
    & df["G3"].between(min_grade, max_grade)
].copy()

if df.empty:
    st.warning("No students match the selected filters.")
    st.stop()

# ---------------- KPI METRICS ----------------
total_students = len(df)
average_grade = df["G3"].mean()
passed_students = int((df["G3"] >= 10).sum())
failed_students = total_students - passed_students
pass_rate = passed_students / total_students * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", f"{total_students:,}")
col2.metric("Average Final Grade", f"{average_grade:.2f}/20")
col3.metric("Students Passed", f"{passed_students:,}")
col4.metric("Pass Rate", f"{pass_rate:.1f}%")

st.divider()

# ---------------- CHART 1: GRADE DISTRIBUTION ----------------
st.subheader("1. Distribution of Final Grades")

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df["G3"], bins=20, kde=True, ax=ax)
ax.set_xlabel("Final Grade (G3)")
ax.set_ylabel("Number of Students")
st.pyplot(fig)
plt.close(fig)

# ---------------- CHART 2: STUDY TIME ----------------
st.subheader("2. Average Grade by Study Time")

study_avg = (
    df.groupby("studytime", as_index=False)["G3"]
    .mean()
    .rename(columns={"G3": "Average Final Grade"})
)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    data=study_avg,
    x="studytime",
    y="Average Final Grade",
    ax=ax
)
ax.set_xlabel("Study Time Category")
ax.set_ylabel("Average Final Grade")
ax.set_ylim(0, 20)
st.pyplot(fig)
plt.close(fig)

if not study_avg.empty:
    best_study = study_avg.loc[
        study_avg["Average Final Grade"].idxmax()
    ]
    st.info(
        f"Study-time category {int(best_study['studytime'])} "
        f"has the highest average grade in this selection: "
        f"{best_study['Average Final Grade']:.2f}/20."
    )

# ---------------- CHART 3: PAST FAILURES ----------------
st.subheader("3. Average Grade by Past Failures")

failure_avg = (
    df.groupby("failures", as_index=False)["G3"]
    .mean()
    .rename(columns={"G3": "Average Final Grade"})
)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(
    data=failure_avg,
    x="failures",
    y="Average Final Grade",
    ax=ax
)
ax.set_xlabel("Number of Past Failures")
ax.set_ylabel("Average Final Grade")
ax.set_ylim(0, 20)
st.pyplot(fig)
plt.close(fig)

# ---------------- CHART 4: ABSENCES ----------------
st.subheader("4. Average Grade by Absence Level")

df["absence_level"] = pd.cut(
    df["absences"],
    bins=[-1, 5, 10, 20, float("inf")],
    labels=["0–5", "6–10", "11–20", "Above 20"]
)

absence_avg = (
    df.groupby("absence_level", observed=False)["G3"]
    .mean()
    .dropna()
)

fig, ax = plt.subplots(figsize=(8, 4))
absence_avg.plot(kind="bar", ax=ax)
ax.set_xlabel("Number of Absences")
ax.set_ylabel("Average Final Grade")
ax.set_ylim(0, 20)
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

# ---------------- CHART 5: CORRELATION HEATMAP ----------------
st.subheader("5. Correlation Between Student Performance Factors")

numeric_columns = [
    c for c in [
        "age", "Medu", "Fedu", "studytime",
        "failures", "absences", "G1", "G2", "G3"
    ]
    if c in df.columns
]

if len(numeric_columns) >= 2:
    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Not enough numeric columns for a correlation heatmap.")

# ---------------- STUDENT DATA EXPLORER ----------------
st.subheader("6. Student Data Explorer")

search = st.text_input(
    "Search student records by school or gender"
)

display_df = df.copy()

if search:
    search = search.strip().lower()
    display_df = display_df[
        display_df.astype(str).apply(
            lambda col: col.str.lower().str.contains(
                search, regex=False, na=False
            )
        ).any(axis=1)
    ]

st.write(f"Records displayed: {len(display_df)}")
st.dataframe(display_df, use_container_width=True)

csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv_data,
    file_name="filtered_student_performance.csv",
    mime="text/csv"
)

# ---------------- KEY INSIGHTS ----------------
st.divider()
st.header("📌 Key Insights and Recommendations")

st.subheader("1. Overall Performance")

k1, k2, k3 = st.columns(3)
k1.metric("Students Passed", passed_students)
k2.metric("Students Failed", failed_students)
k3.metric("Average Grade", f"{average_grade:.2f}/20")

st.write(
    f"Of the {total_students} students in the current selection, "
    f"{passed_students} passed and {failed_students} did not pass. "
    f"The pass rate is {pass_rate:.1f}%."
)

st.divider()
st.subheader("2. Study Time Analysis")

if not study_avg.empty:
    st.dataframe(study_avg, use_container_width=True)

st.divider()
st.subheader("3. Past Failures Analysis")

if not failure_avg.empty:
    st.dataframe(failure_avg, use_container_width=True)

st.divider()
st.subheader("4. Key Findings")

st.markdown(f"""
- **Overall Performance:** The average final grade is
  {average_grade:.2f}/20 in the current selection.
- **Pass Rate:** {pass_rate:.1f}% of selected students achieved
  a final grade of at least 10/20.
- **Study Time:** Compare the study-time categories to explore
  differences in average final grades.
- **Past Failures:** Review historical failures alongside grades
  to identify groups that may need additional support.
- **Attendance:** Compare absence groups to investigate potential
  differences in academic performance.
""")

st.subheader("5. Recommendations")

st.markdown("""
1. Identify students with lower final grades for targeted academic support.
2. Compare study-time groups before drawing conclusions about study habits.
3. Review students with previous failures and consider additional guidance.
4. Examine attendance patterns to identify groups needing further investigation.
5. Use the dashboard filters to compare schools, genders, and grade ranges.
""")

st.caption(
    "Note: These findings describe patterns in the selected data. "
    "They do not establish that study time, attendance, or past failures "
    "caused a particular academic result."
)
