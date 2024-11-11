import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# Załadowanie danych
penguins_not_cleaned = pd.read_csv('data/penguins.csv')

# Czyszczenie danych z wartościami NaN
penguins = penguins_not_cleaned.dropna()

# Wyświetlenie pierwszych kilku wierszy
print(penguins.head())

# Wyświetlenie statystyk danych
print(penguins.describe())

# Wykresy rozkładu cech
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(penguins['CulmenLength'], ax=axes[0, 0])
sns.histplot(penguins['CulmenDepth'], ax=axes[0, 1])
sns.histplot(penguins['FlipperLength'], ax=axes[1, 0])
sns.histplot(penguins['BodyMass'], ax=axes[1, 1])

plt.tight_layout()
plt.show()

# Analiza korelacji między cechami
correlation_matrix = penguins[['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']].corr()
print(correlation_matrix)

# Wykres korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korelacja między cechami')
plt.show()

# Analiza rozkładu gatunków
species_counts = penguins['Species'].value_counts()
print(species_counts)

# Wykres rozkładu gatunków
plt.figure(figsize=(10, 6))
species_counts.plot(kind='bar')
plt.title('Rozkład gatunków pingwinów')
plt.xlabel('Gatunek')
plt.ylabel('Liczba')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Przygotowanie danych
X = penguins[['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']]
y = penguins['Species']

# Podział danych na zbiory treningowe i walidacyjne
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model sąsiada najbliższego
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"MSE dla modelu sąsiada najbliższego: {mse_knn}")

#Drzewo decyzyjne
model_tree = DecisionTreeClassifier(max_depth=3)
model_tree.fit(X_train, y_train)
y_pred_rf = model_tree.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"MSE dla modelu Drzewo decyzyjne: {mse_rf}")

# # Model SVC
model_svm = SVC(kernel='rbf', probability=True)
model_svm.fit(X_train, y_train)
y_pred_svr = model_svm.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"MSE dla modelu SVC: {mse_svr}")

# Model losowego lasu classifier
model_forest = RandomForestClassifier(n_estimators=1000, max_depth=3)
model_forest.fit(X_train, y_train)
y_pred_rf = model_forest.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"MSE dla modelu losowego lasu classifier: {mse_rf}")

# Model losowego lasu regression
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"MSE dla modelu losowego lasu regrresion: {mse_rf}")

#AdaBoost
model_adaboost = AdaBoostClassifier(n_estimators=50)
model_adaboost.fit(X_train, y_train)
y_pred_dt = model_adaboost.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"MSE dla modelu AdaBoost: {mse_dt}")

model_voting = VotingClassifier(estimators=[('Tree', model_tree),
                                            ('Random Forest', model_forest),
                                            ('AdaBoost', model_adaboost)],
                                voting='soft')

model_voting.fit(X_train, y_train)

# Klasyfikacja na podstawie wielu zmiennych
random_forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
params_rf = {'max_depth': [3, 5, 10, 20],
             'min_samples_leaf': [3, 5, 10, 15]}
rf_gridsearch = GridSearchCV(random_forest,
                             params_rf,
                             scoring='f1_macro',
                             cv=5,
                             verbose=10, n_jobs=-1)
rf_gridsearch.fit(X_train, y_train)
print('\nBest hyperparameter:', rf_gridsearch.best_params_)
rf_model_v2 = rf_gridsearch.best_estimator_

# Model liniowy
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"MSE dla modelu liniowego: {mse_lr}")

# Model losowego lasu
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"MSE dla modelu losowego lasu: {mse_rf}")

# Model SVR
model_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
model_svr.fit(X_train_scaled, y_train)
y_pred_svr = model_svr.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"MSE dla modelu SVR: {mse_svr}")

# Model decyzji drzewa
model_dt = DecisionTreeRegressor(max_depth=10, random_state=42)
model_dt.fit(X_train_scaled, y_train)
y_pred_dt = model_dt.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"MSE dla modelu decyzji drzewa: {mse_dt}")

# Model Bayesa
model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)
y_pred_nb = model_nb.predict(X_test_scaled)
accuracy_nb = model_nb.score(X_test_scaled, y_test)
print(f"Dokładność dla modelu Bayesa: {accuracy_nb}")
