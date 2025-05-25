## Exploração Inicial dos Dados

```python
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
display(df.head())

print("\nInformações sobre o dataset:")
display(df.info())

print("\nEstatísticas descritivas:")
display(df.describe())

print("\nValores faltantes por coluna:")
display(df.isnull().sum())

print("\nValores duplicados:", df.duplicated().sum())
```

## Limpeza e Preparação dos Dados

```python
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
```

![01](https://github.com/user-attachments/assets/736a9059-ef7c-488f-82f5-1f9e3a5a4d0e)

## Análise Exploratória

#### Distribuição do Tempo de Entrega

```python
fig = px.histogram(df, x='Delivery_Time', nbins=50, 
                   title='Distribuição do Tempo de Entrega',
                   labels={'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()
```

![01](https://github.com/user-attachments/assets/255e593e-3d46-46b0-84e5-133720452c4a)

#### Tempo de Entrega por Tipo de Veículo

```python
fig = px.box(df, x='Vehicle', y='Delivery_Time', 
             title='Tempo de Entrega por Tipo de Veículo',
             labels={'Vehicle': 'Tipo de Veículo', 'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()
```

![01](https://github.com/user-attachments/assets/d7fac505-a03c-4e76-a263-7c4987f70d12)

#### Impacto das Condições Climáticas

```python
fig = px.box(df, x='Weather', y='Delivery_Time', 
             title='Impacto das Condições Climáticas no Tempo de Entrega',
             labels={'Weather': 'Condição Climática', 'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()
```

![01](https://github.com/user-attachments/assets/40807084-ab24-41f3-9f05-06b832e8dc35)

#### Relação entre Distância e Tempo de Entrega

```python
fig = px.scatter(df, x='Distance', y='Delivery_Time', 
                 trendline='ols',
                 title='Relação entre Distância e Tempo de Entrega',
                 labels={'Distance': 'Distância (km)', 'Delivery_Time': 'Tempo de Entrega (minutos)'})
fig.show()
```

![01](https://github.com/user-attachments/assets/1a47709b-f8cd-40a2-ba47-16c797ffba1e)

#### Tempo Médio de Entrega por Área

```python
area_avg = df.groupby('Area')['Delivery_Time'].mean().reset_index()
fig = px.bar(area_avg, x='Area', y='Delivery_Time',
             title='Tempo Médio de Entrega por Área',
             labels={'Area': 'Área', 'Delivery_Time': 'Tempo Médio de Entrega (minutos)'})
fig.show()
```

![01](https://github.com/user-attachments/assets/b80df64b-3b31-44f1-bbbc-25eb37674410)






