import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Load the CSV file into a Pandas DataFrame
file_path = "City_of_Seattle_Wage_Data.csv"
wage_data_df = pd.read_csv(file_path)

# Create the histogram for hourly wages
histogram = px.histogram(
    wage_data_df, x="Hourly Rate ", nbins=50, title="Distribution of Hourly Wages"
)

# Create the bar chart for top 10 employees with the highest hourly wages
top_10_highest_wages = wage_data_df.nlargest(10, "Hourly Rate ")
bar_chart_top_10 = px.bar(
    top_10_highest_wages,
    x="Job Title",
    y="Hourly Rate ",
    text="Hourly Rate ",
    title="Top 10 Employees with the Highest Hourly Wages",
    labels={"Job Title": "Employee Job Title"},
)

# Create the bar chart for mean hourly wages by department
mean_wages_by_department = (
    wage_data_df.groupby("Department")["Hourly Rate "]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
bar_chart_departments = px.bar(
    mean_wages_by_department,
    x="Department",
    y="Hourly Rate ",
    text="Hourly Rate ",
    title="Distribution of Mean Hourly Wages by Department",
    labels={"Department": "Department"},
)

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(
    children=[
        html.H1(children="City of Seattle Wage Data Dashboard"),
        html.Div(
            children="""An interactive dashboard to visualize the distribution of hourly wages for employees in the City of Seattle."""
        ),
        dcc.Graph(id="histogram", figure=histogram),
        dcc.Graph(id="bar_chart_top_10", figure=bar_chart_top_10),
        dcc.Graph(id="bar_chart_departments", figure=bar_chart_departments),
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
