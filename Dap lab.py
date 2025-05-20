import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hotel_reviews_sentiment.csv")

# DATA PREPROCESSING
# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Rename columns
df.rename(columns={
    "review": "Review",
    "sentiment": "Sentiment",
    "rating": "Rating"
}, inplace=True)

# Drop nulls if any
df.dropna(inplace=True)

# Add Review Length column
df["Review_Length"] = df["Review"].apply(lambda x: len(str(x).split()))

# DATA ORGANIZING
print("\n Data Shape:", df.shape)
print("\n Sentiment Counts:\n", df["Sentiment"].value_counts())
print("\n Rating Stats:\n", df["Rating"].describe())
print("\n Review Length Stats:\n", df["Review_Length"].describe())

# Grouped summary
grouped = df.groupby("Sentiment").agg({
    "Rating": ["mean", "std", "count"],
    "Review_Length": ["mean", "std"]
}).round(2)
print("\n Grouped Statistics by Sentiment:\n", grouped)

# VISUALIZATION
# Sentiment Count Plot
sentiment_counts = df["Sentiment"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Rating Histogram
plt.figure(figsize=(6, 4))
plt.hist(df["Rating"], bins=10, color='skyblue', edgecolor='black')
plt.title("Rating Distribution")
plt.xlabel("Rating (out of 5)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot of Rating by Sentiment
plt.figure(figsize=(6, 4))
data_to_plot = [df[df["Sentiment"] == s]["Rating"] for s in ["Positive", "Neutral", "Negative"]]
plt.boxplot(data_to_plot, labels=["Positive", "Neutral", "Negative"])
plt.title("Rating by Sentiment")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# Review Length Histogram
plt.figure(figsize=(6, 4))
plt.hist(df["Review_Length"], bins=20, color="orange", edgecolor="black")
plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot: Review Length vs Rating
plt.figure(figsize=(6, 4))
colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
for sentiment in df["Sentiment"].unique():
    subset = df[df["Sentiment"] == sentiment]
    plt.scatter(subset["Review_Length"], subset["Rating"],
                label=sentiment, color=colors[sentiment], alpha=0.6)
plt.title("Review Length vs Rating")
plt.xlabel("Review Length (words)")
plt.ylabel("Rating")
plt.legend()
plt.tight_layout()
plt.show()