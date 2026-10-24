
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the mathematics student dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Display the first five rows
print("FIRST FIVE ROWS")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Display basic statistical information
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# FINAL GRADE ANALYSIS

print("\nAVERAGE GRADES")
print("G1 average:", df["G1"].mean())
print("G2 average:", df["G2"].mean())
print("G3 average:", df["G3"].mean())

print("\nAVERAGE FINAL GRADE BY STUDY TIME")
print(df.groupby("studytime")["G3"].mean().round(2))

print("\nAVERAGE FINAL GRADE BY NUMBER OF FAILURES")
print(df.groupby("failures")["G3"].mean().round(2))

print("\nAVERAGE FINAL GRADE BY ABSENCE LEVEL")
df["absence_level"] = pd.cut(
    df["absences"],
    bins=[-1, 5, 10, 20, float("inf")],
    labels=["0-5", "6-10", "11-20", "Above 20"]
)
print(df.groupby("absence_level", observed=False)["G3"].mean().round(2))
# DATA VISUALIZATION

sns.set_theme(style="whitegrid")

# 1. Distribution of Final Grades
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="G3", bins=15, kde=True)
plt.title("Distribution of Students' Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 2. Average Final Grade by Study Time

studytime_grades = df.groupby("studytime")["G3"].mean().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(data=studytime_grades, x="studytime", y="G3")

plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Average Final Grade (G3)")

plt.tight_layout()
plt.show()

# 3. Average Final Grade by Number of Past Failures

failure_grades = df.groupby("failures")["G3"].mean().reset_index()

plt.figure(figsize=(8, 5))

sns.barplot(data=failure_grades, x="failures", y="G3")

plt.title("Average Final Grade by Number of Past Failures")
plt.xlabel("Number of Past Failures")
plt.ylabel("Average Final Grade (G3)")

plt.tight_layout()
plt.show()

# 4. Average Final Grade by Absence Level

absence_grades = df.groupby("absence_level", observed=False)["G3"].mean().reset_index()

plt.figure(figsize=(9, 5))

sns.barplot(data=absence_grades, x="absence_level", y="G3")

plt.title("Average Final Grade by Absence Level")
plt.xlabel("Number of Absences")
plt.ylabel("Average Final Grade (G3)")

plt.tight_layout()
plt.show()

# 5. Correlation Analysis

correlation_columns = [
    "age",
    "Medu",
    "Fedu",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
    "G3"
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Student Performance Factors")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 2: STATISTICAL ANALYSIS
# ==========================================

print("\n" + "=" * 50)
print("STUDENT PERFORMANCE - KEY STATISTICS")
print("=" * 50)

# 1. Total number of students
print("\nTotal Students:", len(df))

# 2. Average final grade
print("Average Final Grade:", round(df["G3"].mean(), 2))

# 3. Highest and lowest final grades
print("Highest Final Grade:", df["G3"].max())
print("Lowest Final Grade:", df["G3"].min())

# 4. Count passed and failed students
passed = (df["G3"] >= 10).sum()
failed = (df["G3"] < 10).sum()

print("Students Passed:", passed)
print("Students Failed:", failed)

# 5. Pass percentage
pass_percentage = (passed / len(df)) * 100
print("Pass Percentage:", round(pass_percentage, 2), "%")

# 6. Average grade by study time
print("\nAverage Grade by Study Time:")
print(df.groupby("studytime")["G3"].mean().round(2))

# 7. Average grade by past failures
print("\nAverage Grade by Past Failures:")
print(df.groupby("failures")["G3"].mean().round(2))
