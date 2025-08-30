# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# df=pd.read_csv("house.csv",delim_whitespace=True,header=None)
# df.columns =[
#     'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'
# ]

# # Quick data checking
# print(df.shape)
# print(df.info())
# print(df.describe())
# # print(df)

# # missing value handling
# print(df.isnull().sum())
# df=df.dropna()

# # outliers removal(z-score method)
# from scipy.stats import zscore
# z_scores=np.abs(zscore(df))
# df=df[(z_scores<3).all(axis=1)]

# # Feature engineering
# df['Price_per_room']=df['MEDV']/df['RM']
# df['Is_old_area']=df['AGE'].apply(lambda x: 1 if x>80 else 0)

# # 1. Histogram of MEDV (Median Home Value)
# plt.figure(figsize=(8,5))
# sns.histplot(df['MEDV'],bins=30, kde=True, color='skyblue')
# plt.title("Distribution of Median Home Value (MEDV)")
# plt.xlabel("MEDV ($1000s)")
# plt.ylabel("Frequency")
# plt.show()

# # 2. Scatter Plot : RM vs MEDV
# plt.figure(figsize=(8,5))
# sns.scatterplot(x="RM", y="MEDV", data=df, color="green")
# plt.title("Average Rooms vs Median Home Value")
# plt.xlabel("Average Rooms (RM)")
# plt.ylabel("MEDV ($1000s)")
# plt.show()

# # 3. Line Plot: TAX vs MEDV
# df_sorted = df.sort_values("TAX")
# plt.figure(figsize=(8,5))
# plt.plot(df_sorted["TAX"], df_sorted["MEDV"], color="red")
# plt.title("Property Tax vs Median Home Value")
# plt.xlabel("Property Tax Rate")
# plt.ylabel("MEDV ($1000s)")
# plt.show()

# # 4. Correlation Heatmap
# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Matrix")
# plt.show()

# # 5. Feature Importance from Linear Regression
# X = df.drop("MEDV", axis=1)
# y = df["MEDV"]

# model = LinearRegression()
# model.fit(X, y)

# importance = pd.Series(model.coef_, index=X.columns)
# plt.figure(figsize=(10,6))
# importance.sort_values().plot(kind="barh", color="purple")
# plt.title("Feature Impact on Median Home Value")
# plt.xlabel("Coefficient Value")
# plt.ylabel("Feature")
# plt.show()

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load cleaned dataset
df = pd.read_csv("house.csv",delim_whitespace=True,header=None)
df.columns =[
     'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'
]
st.title("🏡 Boston Housing Dashboard")

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
min_rm, max_rm = st.sidebar.slider("Average Rooms (RM)", float(df["RM"].min()), float(df["RM"].max()), (5.0, 7.0))
min_age, max_age = st.sidebar.slider("Property Age", float(df["AGE"].min()), float(df["AGE"].max()), (30.0, 90.0))
chas_filter = st.sidebar.selectbox("Near River (CHAS)", options=[0, 1, "All"])

# Apply filters
filtered_df = df[(df["RM"] >= min_rm) & (df["RM"] <= max_rm) & (df["AGE"] >= min_age) & (df["AGE"] <= max_age)]
if chas_filter != "All":
    filtered_df = filtered_df[filtered_df["CHAS"] == chas_filter]

st.subheader("📊 Histogram of Median Home Value")
fig1, ax1 = plt.subplots()
sns.histplot(filtered_df["MEDV"], bins=30, kde=True, color="skyblue", ax=ax1)
st.pyplot(fig1)

st.subheader("📈 Rooms vs Price")
fig2, ax2 = plt.subplots()
sns.scatterplot(x="RM", y="MEDV", data=filtered_df, ax=ax2)
st.pyplot(fig2)

st.subheader("📉 Tax vs Price")
fig3, ax3 = plt.subplots()
df_sorted = filtered_df.sort_values("TAX")
ax3.plot(df_sorted["TAX"], df_sorted["MEDV"], color="red")
st.pyplot(fig3)

st.subheader("🔥 Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(filtered_df.corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

st.subheader("📊 Feature Importance (Linear Regression)")
X = filtered_df.drop("MEDV", axis=1)
y = filtered_df["MEDV"]
model = LinearRegression()
model.fit(X, y)
importance = pd.Series(model.coef_, index=X.columns)
fig5, ax5 = plt.subplots()
importance.sort_values().plot(kind="barh", color="teal", ax=ax5)
st.pyplot(fig5)