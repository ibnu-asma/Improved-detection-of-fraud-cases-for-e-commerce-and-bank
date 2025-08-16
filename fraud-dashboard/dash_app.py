from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import requests
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

# Initialize the Dash app with Bootstrap styling
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set a color palette for consistent styling
color_palette = {
    'background': '#ECF0F1',
    'text': '#2C3E50',
    'box_border': '#3498DB',
    'total_transactions_bg': '#D5E8FD',  # Light Blue
    'total_fraud_cases_bg': '#FADBD8',    # Light Red
    'fraud_percentage_bg': '#D9EAD3',      # Light Green
    'box_text_color': '#34495E',
    'line_chart': '#27AE60',
    'bar_chart_device': '#E74C3C',
    'bar_chart_browser': '#8E44AD'
}

# Define the layout of the dashboard
app.layout = html.Div(style={'backgroundColor': color_palette['background'], 'padding': '0px'}, children=[
    dbc.Container(fluid=False, children=[
        # Header section
        html.Div(style={'textAlign': 'center', 'padding': '20px'}, children=[
            html.H1(children='Fraud Detection Dashboard', style={'color': color_palette['text']}),
            html.Hr(),
        ]),

        # Summary boxes displaying total transactions, fraud cases, and fraud percentage
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(id='total-transactions', style={'color': color_palette['box_text_color']}),
                    html.P('Total Transactions', style={'fontSize': '14px', 'color': '#7F8C8D'}),
                ])
            ], style={
                'border': f'1px solid {color_palette["box_border"]}', 
                'borderRadius': '5px', 
                'backgroundColor': color_palette['total_transactions_bg']
            }), width=4),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(id='total-fraud-cases', style={'color': color_palette['box_text_color']}),
                    html.P('Total Fraud Cases', style={'fontSize': '14px', 'color': '#7F8C8D'}),
                ])
            ], style={
                'border': f'1px solid {color_palette["box_border"]}', 
                'borderRadius': '5px', 
                'backgroundColor': color_palette['total_fraud_cases_bg']
            }), width=4),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H3(id='fraud-percentage', style={'color': color_palette['box_text_color']}),
                    html.P('Fraud Percentage', style={'fontSize': '14px', 'color': '#7F8C8D'}),
                ])
            ], style={
                'border': f'1px solid {color_palette["box_border"]}', 
                'borderRadius': '5px', 
                'backgroundColor': color_palette['fraud_percentage_bg']
            }), width=4),
        ], className="mb-4"),

        # Line chart for fraud trends over time
        dbc.Row([
            dbc.Col(dcc.Graph(id='fraud-trends-chart'), width=12),
        ], className="mb-4"),

        # Geographical analysis of fraud cases
        dbc.Row([
            dbc.Col(dcc.Graph(id='fraud-by-location-chart', style={'height': '600px'}), width=12),
        ], className="mb-4"),

        # Bar charts for fraud cases by device and browser
        dbc.Row([
            dbc.Col(dcc.Graph(id='fraud-by-device-chart'), width=6),
            dbc.Col(dcc.Graph(id='fraud-by-browser-chart'), width=6),
        ])
    ])
])

# Callbacks for updating summary data
@app.callback(
    [Output('total-transactions', 'children'),
     Output('total-fraud-cases', 'children'),
     Output('fraud-percentage', 'children')],
    Input('total-transactions', 'id')  # Dummy input to trigger callback
)
def update_summary(_):
    """Fetch and update summary data from the API."""
    try:
        response = requests.get('http://127.0.0.1:5000/api/fraud_summary')
        response.raise_for_status()  # Raise an error for bad responses
        summary_data = response.json()
        return (
            f'Total Transactions: {summary_data["total_transactions"]}',
            f'Total Fraud Cases: {summary_data["total_fraud_cases"]}',
            f'Fraud Percentage: {summary_data["fraud_percentage"]:.2f}%'
        )
    except Exception as e:
        return ('Error fetching data',) * 3  # Return error message for all outputs

# Callback for fraud trends chart
@app.callback(
    Output('fraud-trends-chart', 'figure'),
    Input('fraud-trends-chart', 'id')  # Dummy input to trigger callback
)
def update_fraud_trends(_):
    """Fetch and update the fraud trends line chart."""
    try:
        response = requests.get('http://127.0.0.1:5000/api/fraud_trends')
        response.raise_for_status()
        trend_data = pd.DataFrame(response.json())
        fig = px.line(
            trend_data,
            x='purchase_date',
            y='fraud_cases',
            title='Fraud Cases Over Time',
            line_shape='spline',
            markers=True,
            template='plotly_white',
            color_discrete_sequence=[color_palette['line_chart']]
        )
        fig.update_traces(hovertemplate='Date: %{x}<br>Fraud Cases: %{y}')
        return fig
    except Exception as e:
        return px.line(title="Error loading data.")  # Return an empty line chart on error

# Callback for fraud cases by location chart
@app.callback(
    Output('fraud-by-location-chart', 'figure'),
    Input('fraud-by-location-chart', 'id')  # Dummy input to trigger callback
)
def update_fraud_by_location(_):
    """Fetch and update the fraud by location choropleth map."""
    try:
        response = requests.get('http://127.0.0.1:5000/api/fraud_by_location')
        response.raise_for_status()
        location_data = pd.DataFrame(response.json())
        fig = px.choropleth(
            location_data,
            locations='country',
            locationmode='country names',
            color='fraud_cases',
            title='Fraud by Location',
            color_continuous_scale='Blues',
            template='plotly_white',
            labels={'fraud_cases': 'Number of Fraud Cases'}
        )
        fig.update_geos(fitbounds="locations", visible=False)
        return fig
    except Exception as e:
        return px.choropleth(title="Error loading data.")  # Return an empty map on error

# Callback for fraud cases by device and browser charts
@app.callback(
    [Output('fraud-by-device-chart', 'figure'),
     Output('fraud-by-browser-chart', 'figure')],
    Input('fraud-by-device-chart', 'id')  # Dummy input to trigger callback
)
def update_fraud_by_device_browser(_):
    """Fetch and update the fraud cases by device and browser bar charts."""
    try:
        response = requests.get('http://127.0.0.1:5000/api/fraud_by_device_browser')
        response.raise_for_status()
        device_browser_data = pd.DataFrame(response.json())

        # Get top devices and browsers
        top_devices = device_browser_data.nlargest(10, 'fraud_cases')
        top_browsers = device_browser_data.groupby('browser').agg({'fraud_cases': 'sum'}).reset_index()
        top_browsers = top_browsers.nlargest(10, 'fraud_cases')

        # Create the device bar chart
        device_fig = px.bar(
            top_devices,
            x='device_id',
            y='fraud_cases',
            title='Top Devices by Fraud Cases',
            labels={'device_id': 'Device', 'fraud_cases': 'Number of Fraud Cases'},
            template='plotly_white',
            color_discrete_sequence=[color_palette['bar_chart_device']]
        )
        device_fig.update_traces(hovertemplate='Device: %{x}<br>Fraud Cases: %{y}')

        # Create the browser bar chart
        browser_fig = px.bar(
            top_browsers,
            x='browser',
            y='fraud_cases',
            title='Top Browsers by Fraud Cases',
            labels={'browser': 'Browser', 'fraud_cases': 'Number of Fraud Cases'},
            template='plotly_white',
            color_discrete_sequence=[color_palette['bar_chart_browser']]
        )
        browser_fig.update_traces(hovertemplate='Browser: %{x}<br>Fraud Cases: %{y}')

        return device_fig, browser_fig
    except Exception as e:
        return (px.bar(title="Error loading device data."), px.bar(title="Error loading browser data."))  # Return empty charts on error

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
