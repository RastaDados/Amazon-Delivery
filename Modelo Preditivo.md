## Modelagem Preditiva

#### Codificando as variáveis categóricas

```python
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
```

#### Modelo de Regressão Linear

```python
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print("Desempenho da Regressão Linear:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

![01](https://github.com/user-attachments/assets/6a72a092-9054-4a20-8397-333c77d268a9)

#### Modelo Random Forest

```python
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
```

![01](https://github.com/user-attachments/assets/fffbc029-ca0d-4d23-a37b-5271b9f98655)

![01](https://github.com/user-attachments/assets/54445fd9-4a7a-465e-a17a-bc3b21377f64)




