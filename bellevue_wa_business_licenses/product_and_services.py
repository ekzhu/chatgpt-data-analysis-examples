import dash
from dash import html, dcc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px

# Load the CSV file into a Pandas DataFrame
file_path = "Business_Licenses_(All).csv"
df = pd.read_csv(file_path)


# Define a function to preprocess the text
def preprocess_text(text):
    # Convert to lowercase and remove non-alphanumeric characters
    return " ".join(word.lower() for word in text.split() if word.isalnum())


# Preprocess the 'ProductsAndServices' column and remove missing values
df_clean = df.dropna(subset=["ProductsAndServices"])
df_clean["ProductsAndServices"] = df_clean["ProductsAndServices"].apply(preprocess_text)

# Use TF-IDF vectorizer to extract features from the text
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df_clean["ProductsAndServices"])

# Perform k-means clustering to group similar products and services
num_clusters = 10  # Number of clusters (broader categories)
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
df_clean["CategoryCluster"] = kmeans.labels_


# Extract the top terms in each cluster
def extract_top_terms_in_cluster(cluster_num, vectorizer, kmeans, top_n=5):
    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    # Get the top terms in the specified cluster
    top_terms_indices = centroids[cluster_num].argsort()[-top_n:][::-1]
    top_terms = [vectorizer.get_feature_names_out()[i] for i in top_terms_indices]
    return top_terms


# Suggest labels based on the top terms in each cluster
suggested_labels = {}
for cluster_num in range(num_clusters):
    top_terms = extract_top_terms_in_cluster(cluster_num, vectorizer, kmeans, top_n=5)
    suggested_label = " / ".join(top_terms)
    suggested_labels[cluster_num] = suggested_label

# Update the 'CategoryLabel' column with the suggested labels
df_clean["CategoryLabel"] = df_clean["CategoryCluster"].map(suggested_labels)

df_clean["IssueDate"] = pd.to_datetime(df_clean["IssueDate"])

# Re-plot the trend of businesses opened for each category over time
df_grouped = (
    df_clean.groupby(["IssueDate", "CategoryLabel"])
    .size()
    .unstack()
    .resample("M")
    .sum()
)

# Plot the trend of businesses opened for each category over time (aggregated by month)
df_grouped_long = df_grouped.reset_index().melt(
    id_vars="IssueDate", var_name="Category", value_name="Count"
)
fig1 = px.line(
    df_grouped_long,
    x="IssueDate",
    y="Count",
    color="Category",
    title="Trend of Businesses Opened for Different Categories Over Time",
)

# Plot the average count of businesses opened based on FirstActivityDate, aggregated by month, over the month of year
df_clean["FirstActivityDate"] = pd.to_datetime(
    df_clean["FirstActivityDate"], errors="coerce"
)
df_clean["MonthOfYear"] = df_clean["FirstActivityDate"].dt.month
df_average_count_by_month = (
    df_clean.groupby("MonthOfYear").size().reset_index(name="AverageCount")
)
fig2 = px.bar(
    df_average_count_by_month,
    x="MonthOfYear",
    y="AverageCount",
    title="Average Count of Businesses Opened by First Activity Date (Aggregated by Month, Over Month of Year)",
)

# Convert the 'NAICS' column to string and remove any leading/trailing whitespace
df_clean["NAIC"] = df_clean["Naic"].astype(str).str.strip()

# Group the data by 'IssueDate' and 'NAIC', and count the number of businesses opened
df_naics_grouped = (
    df_clean.groupby(["IssueDate", "NAIC"]).size().unstack().resample("M").sum()
)

# Convert the DataFrame to a long format for Plotly
df_naics_grouped_long = df_naics_grouped.reset_index().melt(
    id_vars="IssueDate", var_name="NAIC", value_name="Count"
)

# Create a line chart using Plotly Express to show the trend of businesses opened for each NAICS code
fig3 = px.line(
    df_naics_grouped_long,
    x="IssueDate",
    y="Count",
    color="NAIC",
    title="Time Series Trends of Businesses Opened for Each NAICS Code (Aggregated by Month)",
    labels={"NAIC": "NAICS Code"},
)

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the Dash app
app.layout = html.Div(
    [
        html.H1("Business License Analysis Dashboard"),
        dcc.Graph(id="line-chart-1", figure=fig1),
        dcc.Graph(id="bar-chart-2", figure=fig2),
        dcc.Graph(id="line-chart-3", figure=fig3),
    ]
)

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
