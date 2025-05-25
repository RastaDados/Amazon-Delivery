import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

#Carregando os dados
df = pd.read_csv('dados/amazon_delivery.csv')

#Visualização inicial
print("Primeiras linhas do dataset:")
print(df.head())

print("\nInformações sobre o dataset:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nValores faltantes por coluna:")
print(df.isnull().sum())

print("\nValores duplicados:", df.duplicated().sum())

#Tratamento de valores nulos
df['Agent_Rating'].fillna(df['Agent_Rating'].median(), inplace=True)

#Tratando valores NaN
df['Order_Time'] = pd.to_datetime(df['Order_Time'], errors='coerce')



#Criando as features de distância 
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


#Verificando as correlações
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()


fig = px.histogram(df, x='Delivery_Time', nbins=50, 
                   title='Distribuição do Tempo de Entrega',
                   labels={'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()


fig = px.box(df, x='Vehicle', y='Delivery_Time', 
             title='Tempo de Entrega por Tipo de Veículo',
             labels={'Vehicle': 'Tipo de Veículo', 'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()


fig = px.box(df, x='Weather', y='Delivery_Time', 
             title='Impacto das Condições Climáticas no Tempo de Entrega',
             labels={'Weather': 'Condição Climática', 'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()


fig = px.scatter(df, x='Distance', y='Delivery_Time', 
                 trendline='ols',
                 title='Relação entre Distância e Tempo de Entrega',
                 labels={'Distance': 'Distância (km)', 'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()


area_avg = df.groupby('Area')['Delivery_Time'].mean().reset_index()
fig = px.bar(area_avg, x='Area', y='Delivery_Time',
             title='Tempo Médio de Entrega por Área',
             labels={'Area': 'Área', 'Delivery_Time': 'Tempo Médio de Entrega (minutos)'})
fig.show()



#Codificando as variáveis categóricas
cat_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

#Selecionando as features e target
X = df.drop(['Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time', 'Delivery_Time'], axis=1)
y = df['Delivery_Time']

#Dividindo em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print("Desempenho da Regressão Linear:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")



rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\nDesempenho do Random Forest:")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf)}")
print(f"R²: {r2_score(y_test, y_pred_rf)}")

#Importância das features
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

fig = px.bar(feature_importance, x='Importance', y='Feature', 
             title='Importância das Features no Modelo Random Forest')
fig.show()