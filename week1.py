# 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2: Load Dataset
df = pd.read_csv("titanic.csv")

# 3: Initial Inspection
print("Shape of dataset:", df.shape)
print(df.info())
print(df.describe())
print(df.head())

# 4: Check Missing Values
print("\nMissing Values:\n", df.isnull().sum().sort_values(ascending=False))

# 5: Data Cleaning
df = df.drop(columns=["Cabin"])   # Drop Cabin due to too many missing values
df["Age"].fillna(df["Age"].median(), inplace=True)   # Fill Age with median
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # Fill Embarked with mode

# 6: Univariate Analysis
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# 7: Bivariate Analysis
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived")
plt.title("Age vs Fare Colored by Survival")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Embarked", hue="Survived")
plt.title("Survival by Embarkation Port")
plt.show()

# 8: Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 9: Feature Engineering
df["FamilySize"] = df["SibSp"] + df["Parch"]
df["IsChild"] = df["Age"].apply(lambda x: 1 if x < 18 else 0)
sns.countplot(data=df, x="IsChild", hue="Survived")
plt.title("Survival by Child/Adult")
plt.show()

# 10: Deeper Insights

# Gender Survival Rate
print("\nSurvival Rate by Gender:\n", df.groupby("Sex")["Survived"].mean())

# Class Survival Rate
print("\nSurvival Rate by Pclass:\n", df.groupby("Pclass")["Survived"].mean())

# Average Age of Survivors vs Non-Survivors
print("\nAverage Age by Survival:\n", df.groupby("Survived")["Age"].mean())

# Family Size vs Survival
plt.figure(figsize=(8,4))
sns.barplot(x="FamilySize", y="Survived", data=df)
plt.title("Survival by Family Size")
plt.show()

# Fare vs Survival
print("\nAverage Fare by Survival:\n", df.groupby("Survived")["Fare"].mean())

# Embarked vs Survival
print("\nSurvival Rate by Embarked:\n", df.groupby("Embarked")["Survived"].mean())

# Titles from Name
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace(["Mme"], "Mrs")
print("\nSurvival Rate by Title:\n", df.groupby("Title")["Survived"].mean())

# Children Survival Rate
print("\nChildren (<12 yrs) Survival Rate:", df[df["Age"] < 12]["Survived"].mean())

# Ticket Group Size
ticket_counts = df["Ticket"].value_counts()
df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)
plt.figure(figsize=(8,4))
sns.barplot(x="TicketGroupSize", y="Survived", data=df)
plt.title("Survival by Ticket Group Size")
plt.show()

# Combination of Gender & Class
print("\nSurvival by Gender & Class:\n", 
      pd.crosstab(df["Pclass"], df["Sex"], values=df["Survived"], aggfunc="mean"))
