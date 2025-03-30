"""
📊 Data Visualization Dashboard (Python + Dash + Plotly)


👉 यह प्रोजेक्ट Dash + Plotly का उपयोग करके एक Interactive Data Dashboard बनाएगा।
👉 Real-time Data Visualization & Multiple Charts
👉 Resume में दिखाने के लिए Best Project! 🚀

🔹 Step-by-Step Implementation
✅ Step 1: Data Load करें
✅ Step 2: Dash Layout Design करें
✅ Step 3: Dynamic Graphs & Filters जोड़ें
✅ Step 4: Flask Server से Run करें
"""



import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Sample Dataset (CSV File)
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv")

# Show Data Sample
print(df.head())


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("🌍 Data Visualization Dashboard", style={'text-align': 'center'}),
    
    # Dropdown Filter for Country Selection
    dcc.Dropdown(id="country_dropdown",
                 options=[{"label": country, "value": country} for country in df["country"].unique()],
                 multi=False,
                 value="India",
                 style={"width": "50%"}),

    # Line Chart
    dcc.Graph(id="line_chart"),

    # Bar Chart
    dcc.Graph(id="bar_chart"),
])



# Line Chart Callback (GDP Over Time)
@app.callback(
    Output("line_chart", "figure"),
    [Input("country_dropdown", "value")]
)
def update_line_chart(selected_country):
    filtered_df = df[df["country"] == selected_country]
    fig = px.line(filtered_df, x="year", y="gdpPercap", title=f"{selected_country} - GDP Over Time")
    return fig

# Bar Chart Callback (Life Expectancy Over Time)
@app.callback(
    Output("bar_chart", "figure"),
    [Input("country_dropdown", "value")]
)
def update_bar_chart(selected_country):
    filtered_df = df[df["country"] == selected_country]
    fig = px.bar(filtered_df, x="year", y="lifeExp", title=f"{selected_country} - Life Expectancy")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
