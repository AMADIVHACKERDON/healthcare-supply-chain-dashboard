import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv(r"C:\Users\VP\Downloads\Google-Playstore(cleaned_dataset).csv")

# Convert installs to numeric
df["Installs"] = df["Installs"].str.replace("[+,]", "", regex=True).astype(float)

# Sidebar Filters
st.sidebar.header("Filters")
selected_year = st.sidebar.selectbox("Select Release Year", ["All"] + sorted(df["Released"].dropna().unique().tolist()))
selected_category = st.sidebar.selectbox("Select Category", ["All"] + df["Category"].unique().tolist())

# Apply Filters
if selected_year != "All":
    df = df[df["Released"] == selected_year]
if selected_category != "All":
    df = df[df["Category"] == selected_category]

# App Overview Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Apps", f"{len(df):,}")
col2.metric("Categories", df["Category"].nunique())
col3.metric("Total Installs", f"{df['Installs'].sum()/1e9:.1f}bn")
col4.metric("Avg Rating", f"{df['Rating'].mean():.2f}")

# Ratings & Reviews per App
st.subheader("Ratings and Rating Count Per Apps")
top_apps = df.nlargest(10, "Rating")
fig1 = px.bar(top_apps, x="App Name", y="Rating", text="Rating Count", color="Rating")
st.plotly_chart(fig1)

# Category Distribution
st.subheader("Category by App Name")
category_count = df["Category"].value_counts().reset_index()
fig2 = px.bar(category_count, x=category_count.index, y="count", labels={"index": "Category", "count": "Count"})
st.plotly_chart(fig2)

# In-App Purchases Ratio
st.subheader("In-App Purchases Ratio")
in_app_purchases = df["In App Purchases"].value_counts(normalize=True) * 100
fig3 = px.pie(names=in_app_purchases.index, values=in_app_purchases.values, title="In-App Purchases")
st.plotly_chart(fig3)

# Sentiment Analysis by Category
st.subheader("Average Sentiment Score by Category")
sentiment_scores = df.groupby("Category")["Sentiment_Score"].mean().nlargest(5).reset_index()
fig4 = px.bar(sentiment_scores, x="Category", y="Sentiment_Score", color="Sentiment_Score")
st.plotly_chart(fig4)

# App Update Frequency Over Time
st.subheader("Average App Update Frequency Over Time")
update_freq = df.groupby("Release_Year")["Installs"].count().reset_index()
fig5 = px.line(update_freq, x="Release_Year", y="Installs", markers=True)
st.plotly_chart(fig5)

# Top Categories by Rating
st.subheader("Categories By Rating")
top_rated_categories = df.groupby("Category")["Rating"].mean().nlargest(10).reset_index()
st.dataframe(top_rated_categories)
