import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np

#Carregar e preparando os dados 
df = pd.read_csv('dados/amazon_delivery.csv')


#Tratamento de valores nulos
df['Agent_Rating'].fillna(df['Agent_Rating'].median(), inplace=True)

#Tratando valores NaN
df['Order_Time'] = pd.to_datetime(df['Order_Time'], errors='coerce')


#Criando as feature de distância 
def haversine(lat1, lon1, lat2, lon2):
    # Fórmula simplificada para cálculo de distância
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = 6371 * c
    return km

df['Distance'] = df.apply(lambda x: haversine(x['Store_Latitude'], x['Store_Longitude'], 
                                             x['Drop_Latitude'], x['Drop_Longitude']), axis=1)



#Inicializando o app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

#Layout do dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Análise de Entregas Amazon", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Selecione a Área:"),
            dcc.Dropdown(
                id='area-dropdown',
                options=[{'label': area, 'value': area} for area in df['Area'].unique()],
                value=df['Area'].unique(),
                multi=True
            )
        ], width=4),
        
        dbc.Col([
            html.Label("Selecione o Veículo:"),
            dcc.Dropdown(
                id='vehicle-dropdown',
                options=[{'label': vehicle, 'value': vehicle} for vehicle in df['Vehicle'].unique()],
                value=df['Vehicle'].unique(),
                multi=True
            )
        ], width=4),
        
        dbc.Col([
            html.Label("Selecione a Categoria:"),
            dcc.Dropdown(
                id='category-dropdown',
                options=[{'label': category, 'value': category} for category in df['Category'].unique()],
                value=df['Category'].unique(),
                multi=True
            )
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='delivery-time-hist'), width=6),
        dbc.Col(dcc.Graph(id='vehicle-boxplot'), width=6)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='weather-impact'), width=6),
        dbc.Col(dcc.Graph(id='distance-scatter'), width=6)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='area-comparison'), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(html.Div(id='summary-stats'), width=12)
    ])
], fluid=True)

#Callbacks
@app.callback(
    [Output('delivery-time-hist', 'figure'),
     Output('vehicle-boxplot', 'figure'),
     Output('weather-impact', 'figure'),
     Output('distance-scatter', 'figure'),
     Output('area-comparison', 'figure'),
     Output('summary-stats', 'children')],
    [Input('area-dropdown', 'value'),
     Input('vehicle-dropdown', 'value'),
     Input('category-dropdown', 'value')]
)
def update_dashboard(selected_areas, selected_vehicles, selected_categories):
    #Filtrando os dados baseado nas seleções
    filtered_df = df[
        df['Area'].isin(selected_areas) & 
        df['Vehicle'].isin(selected_vehicles) & 
        df['Category'].isin(selected_categories)
    ]
    
    #Histograma do tempo de entrega
    hist_fig = px.histogram(filtered_df, x='Delivery_Time', nbins=30,
                           title='Distribuição do Tempo de Entrega')
    
    #Boxplot por veículo
    box_fig = px.box(filtered_df, x='Vehicle', y='Delivery_Time',
                    title='Tempo de Entrega por Tipo de Veículo')
    
    #Impacto do clima
    weather_fig = px.box(filtered_df, x='Weather', y='Delivery_Time',
                        title='Impacto das Condições Climáticas')
    
    #Dispersão distância vs tempo
    scatter_fig = px.scatter(filtered_df, x='Distance', y='Delivery_Time',
                            trendline='ols',
                            title='Relação Distância-Tempo de Entrega')
    
    #Comparação por área
    area_fig = px.bar(filtered_df.groupby('Area')['Delivery_Time'].mean().reset_index(),
                     x='Area', y='Delivery_Time',
                     title='Tempo Médio de Entrega por Área')
    
    #Estatísticas resumidas
    avg_time = filtered_df['Delivery_Time'].mean()
    min_time = filtered_df['Delivery_Time'].min()
    max_time = filtered_df['Delivery_Time'].max()
    
    stats = dbc.Card([
        dbc.CardBody([
            html.H4("Estatísticas Resumidas", className="card-title"),
            html.P(f"Tempo médio de entrega: {avg_time:.2f} minutos"),
            html.P(f"Tempo mínimo de entrega: {min_time} minutos"),
            html.P(f"Tempo máximo de entrega: {max_time} minutos"),
            html.P(f"Número de pedidos: {len(filtered_df)}")
        ])
    ])
    
    return hist_fig, box_fig, weather_fig, scatter_fig, area_fig, stats

if __name__ == '__main__':
    app.run_server(debug=True)