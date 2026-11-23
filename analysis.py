import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/VP/Downloads/marketing_campaign_dataset - marketing_campaign_dataset.csv")

print(df.head())


print(df.duplicated().sum())
print(df.columns)

# Ensure Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Extract month
df["Month"] = df["Date"].dt.month

# Find the month with the highest average ROI
monthly_roi = df.groupby("Month")["ROI"].mean()
peak_month = monthly_roi.idxmax()

print(f"Month with highest ROI: {peak_month}")

df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"], errors="coerce")

df["CTR"] = (df["Clicks"] / df["Impressions"]) * 100
df["CPC"] = df["Acquisition_Cost"] / df["Clicks"]

df["Conversion Rate"] = (df["Engagement_Score"] / df["Clicks"]) * 100
df["Revenue"] = (df["ROI"] * df["Acquisition_Cost"]) + df["Acquisition_Cost"] 

# Number of Campaigns
total_campaigns = df["Campaign_ID"].nunique()
print("Total Campaigns:", total_campaigns)

# High Performing Campaign
high_performing_campaigns = df[df["ROI"] > 20]
print(high_performing_campaigns)

# Underperforming Campaign
underperforming_campaigns = df[df["ROI"] < 0]
print(underperforming_campaigns)

# Highest Campaign
top_campaign = df.loc[df["ROI"].idxmax(), ["Campaign_ID", "ROI"]]
print(f"Highest Performing Campaign: {top_campaign['Campaign_ID']} with ROI: {top_campaign['ROI']}")

# Lowest Campaign
bottom_campaign = df.loc[df["ROI"].idxmin(), ["Campaign_ID", "ROI"]]
print(f"Lowest Performing Campaign: {bottom_campaign['Campaign_ID']} with ROI: {bottom_campaign['ROI']}")

# Visualization ROI distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["ROI"], bins=30, kde=True, color="blue")
plt.axvline(x=20, color="red", linestyle="--", label="High ROI Threshold")
plt.title("Distribution of ROI across Campaigns")
plt.xlabel("ROI (%)")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# Boxplot for numerical columns
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[["CTR", "CPC", "Conversion Rate", "ROI"]])
plt.title("Outlier Detection")
plt.show()

# Group by Location
location_performance = df.groupby("Location")["ROI"].mean().reset_index()
print(location_performance)
print(location_performance.nlargest(1, "ROI")) # top 1 largest
print(location_performance.nsmallest(1, "ROI")) # top 1 smallest

# Visualization (Bar Plot) of ROI by Location
plt.figure(figsize=(12, 6))
sns.barplot(x="Location", y="ROI", data=location_performance, palette="viridis")
plt.title("Average ROI by Location")
plt.xlabel("Location")
plt.ylabel("Average ROI (%)")
plt.xticks(rotation=45)
plt.show()

# CTR vs. CPC Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["CPC"], y=df["Conversion_Rate"], hue=df["Channel_Used"], palette="coolwarm")
plt.title("CPC vs Conversion Rate")
plt.xlabel("Cost Per Click (CPC)")
plt.ylabel("Conversion Rate (%)")
plt.legend(title="Channel Used")
plt.show()

# Heatmap for Correlation Between Metrics
plt.figure(figsize=(10, 6))
sns.heatmap(df[["ROI", "CTR", "CPC", "Conversion_Rate"]].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Marketing Metrics")
plt.show()

df_sorted = df.sort_values("ROI", ascending=False).head(10)

#  Campaign Performance by ROI
plt.figure(figsize=(10, 6))
sns.barplot(x="Campaign_ID", y="ROI", hue="Channel_Used", data=df_sorted, palette="viridis")
plt.xticks(rotation=45)
plt.title("Campaign Performance by ROI")
plt.xlabel("Campaign ID")
plt.ylabel("ROI (%)")
plt.legend(title="Channel Used")
plt.show()

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.dropna(subset=["ROI", "Date"])

df = df.dropna(subset=["Channel_Used"])  # Drop rows with missing Channel_Used

# Filter top 5 most frequent channels
top_channels = df["Channel_Used"].value_counts().index[:5]
df_filtered = df[df["Channel_Used"].isin(top_channels)]

# ROI Trend Over Time
plt.figure(figsize=(12,6))
sns.lineplot(x=df_filtered["Date"], y=df_filtered["ROI"], hue=df_filtered["Channel_Used"], style=df_filtered["Channel_Used"], marker="o")
plt.xticks(rotation=45)
plt.title("ROI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("ROI (%)")
plt.legend(title="Channel Used")
plt.show()

