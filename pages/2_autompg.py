# ============================================
# PARTIE 2 : RÉGRESSION (AUTO-MPG)
# ============================================

print("=" * 60)
print("PARTIE 2 : RÉGRESSION - AUTO-MPG")
print("=" * 60)


# Importation des bibliothèques nécessaires

import pandas as pd                  # Pour la manipulation des données
import numpy as np                    # Pour les calculs numériques
import matplotlib.pyplot as plt        # Pour les graphiques de base
import seaborn as sns                  # Pour des graphiques plus sophistiqués
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')      # Pour ignorer les avertissements non critiques
import pickle



# ============================================
# CHARGEMENT ET ANALYSE DES DONNÉES
# ============================================

print("\n" + "=" * 40)
print("CHARGEMENT ET ANALYSE DU DATASET")
print("=" * 40)

# Note : Le dataset auto-mpg n'est pas dans les fichiers fournis
# On va le charger depuis une URL (source UCI)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

# Définir les noms de colonnes
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']

# Charger les données
df_mpg = pd.read_csv(url, names=column_names, na_values='?', 
                      comment='\t', sep='\s+', skipinitialspace=True)

print(f"\nDimensions du dataset : {df_mpg.shape}")
print(f"Nombre d'instances : {df_mpg.shape[0]}")
print(f"Nombre de caractéristiques : {df_mpg.shape[1]}")

print("\nAperçu des 5 premières lignes :")
print(df_mpg.head())

print("\nInformations sur les données :")
print(df_mpg.info())

print("\nStatistiques descriptives :")
print(df_mpg.describe())

# ============================================
# ANALYSE DES VALEURS MANQUANTES
# ============================================

print("\n" + "=" * 40)
print("ANALYSE DES VALEURS MANQUANTES")
print("=" * 40)

missing_values = df_mpg.isnull().sum()
print(f"\nValeurs manquantes par colonne :")
print(missing_values[missing_values > 0])

# Traitement des valeurs manquantes
if missing_values.sum() > 0:
    print("\nTraitement des valeurs manquantes :")
    # Pour 'horsepower', on remplace par la médiane
    df_mpg['horsepower'].fillna(df_mpg['horsepower'].median(), inplace=True)
    print("  ✓ Valeurs manquantes de 'horsepower' remplacées par la médiane")

# Vérification
print(f"\nValeurs manquantes après traitement : {df_mpg.isnull().sum().sum()}")

# ============================================
# ANALYSE EXPLORATOIRE
# ============================================

print("\n" + "=" * 40)
print("ANALYSE EXPLORATOIRE")
print("=" * 40)

# Distribution de la variable cible (mpg)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_mpg['mpg'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('MPG (Miles Per Gallon)')
plt.ylabel('Fréquence')
plt.title('Distribution de la consommation')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df_mpg['mpg'])
plt.ylabel('MPG')
plt.title('Boxplot de la consommation')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mpg_distribution.png', dpi=150)
plt.show()

print(f"\nStatistiques de mpg :")
print(f"   - Min : {df_mpg['mpg'].min():.1f}")
print(f"   - Max : {df_mpg['mpg'].max():.1f}")
print(f"   - Moyenne : {df_mpg['mpg'].mean():.1f}")
print(f"   - Médiane : {df_mpg['mpg'].median():.1f}")

# Matrice de corrélation
plt.figure(figsize=(10, 8))
numeric_cols = df_mpg.select_dtypes(include=[np.number]).columns
correlation_matrix = df_mpg[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.savefig('mpg_correlation.png', dpi=150)
plt.show()

print("\nCorrélations avec mpg :")
corr_with_mpg = correlation_matrix['mpg'].sort_values(ascending=False)
print(corr_with_mpg)

# ============================================
# PRÉPARATION DES DONNÉES
# ============================================

print("\n" + "=" * 40)
print("PRÉPARATION DES DONNÉES POUR LA RÉGRESSION")
print("=" * 40)

# Sélectionner les features (on exclut 'car_name' comme indiqué dans l'énoncé)
feature_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'model_year', 'origin']
X = df_mpg[feature_cols].copy()
y = df_mpg['mpg'].copy()

print(f"\nFeatures sélectionnées : {feature_cols}")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

# Encoder 'origin' (variable catégorielle)
print("\nEncodage de la variable 'origin' :")
print(df_mpg['origin'].value_counts().sort_index())
# Origin est déjà numérique (1,2,3), mais on pourrait la considérer comme catégorielle
# On va la garder comme numérique pour l'instant

# Normalisation (cruciale pour KNN)
scaler_reg = StandardScaler()
X_scaled = scaler_reg.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

print("\nStatistiques après normalisation :")
print(X_scaled.describe().round(2))

# Split train/test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTaille train : {X_train_reg.shape[0]}")
print(f"Taille test : {X_test_reg.shape[0]}")

# ============================================
# RÉGRESSION AVEC KNN
# ============================================

print("\n" + "=" * 40)
print("RÉGRESSION AVEC K-NEAREST NEIGHBORS")
print("=" * 40)

# Créer le modèle KNN pour régression
knn_reg = KNeighborsRegressor(n_neighbors=5)

# Entraînement
knn_reg.fit(X_train_reg, y_train_reg)

# Prédictions
y_train_pred_knn = knn_reg.predict(X_train_reg)
y_test_pred_knn = knn_reg.predict(X_test_reg)

# Calcul des métriques
train_mae = mean_absolute_error(y_train_reg, y_train_pred_knn)
train_mse = mean_squared_error(y_train_reg, y_train_pred_knn)
train_r2 = r2_score(y_train_reg, y_train_pred_knn)

test_mae = mean_absolute_error(y_test_reg, y_test_pred_knn)
test_mse = mean_squared_error(y_test_reg, y_test_pred_knn)
test_r2 = r2_score(y_test_reg, y_test_pred_knn)

print(f"\nPerformances KNN (k=5) :")
print(f"\n  Entraînement :")
print(f"    MAE  (Mean Absolute Error) : {train_mae:.4f}")
print(f"    MSE  (Mean Squared Error)  : {train_mse:.4f}")
print(f"    R²   (Coefficient)         : {train_r2:.4f}")

print(f"\n  Test :")
print(f"    MAE  : {test_mae:.4f}")
print(f"    MSE  : {test_mse:.4f}")
print(f"    R²   : {test_r2:.4f}")

# Interprétation des métriques
print(f"\nInterprétation :")
print(f"   - MAE : en moyenne, l'erreur est de {test_mae:.2f} mpg")
print(f"   - MSE : {test_mse:.2f} (pénalise plus les grosses erreurs)")
print(f"   - R²  : {test_r2:.2%} de la variance expliquée")

# ============================================
# ÉTUDE DE L'INFLUENCE DE k
# ============================================

print("\n" + "=" * 40)
print("INFLUENCE DU PARAMÈTRE k")
print("=" * 40)

# Tester différentes valeurs de k
k_values = range(1, 51)
train_mae_k = []
test_mae_k = []
train_r2_k = []
test_r2_k = []

print("Calcul des performances pour k = 1 à 50...")
for k in k_values:
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X_train_reg, y_train_reg)
    
    y_train_pred = knn_temp.predict(X_train_reg)
    y_test_pred = knn_temp.predict(X_test_reg)
    
    train_mae_k.append(mean_absolute_error(y_train_reg, y_train_pred))
    test_mae_k.append(mean_absolute_error(y_test_reg, y_test_pred))
    train_r2_k.append(r2_score(y_train_reg, y_train_pred))
    test_r2_k.append(r2_score(y_test_reg, y_test_pred))
    
    if k % 10 == 0:
        print(f"  k={k:2d} : MAE test={test_mae_k[-1]:.4f}, R²={test_r2_k[-1]:.4f}")

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, train_mae_k, 'b-', label='Train MAE', linewidth=2)
plt.plot(k_values, test_mae_k, 'r-', label='Test MAE', linewidth=2)
plt.xlabel('k (nombre de voisins)')
plt.ylabel('MAE')
plt.title('Influence de k sur MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_values, train_r2_k, 'b-', label='Train R²', linewidth=2)
plt.plot(k_values, test_r2_k, 'r-', label='Test R²', linewidth=2)
plt.xlabel('k (nombre de voisins)')
plt.ylabel('R²')
plt.title('Influence de k sur R²')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_regression_k_influence.png', dpi=150)
plt.show()

# Trouver le meilleur k
best_k_idx = np.argmin(test_mae_k)
best_k = k_values[best_k_idx]
best_mae = test_mae_k[best_k_idx]
best_r2 = test_r2_k[best_k_idx]

print(f"\nMeilleur k : {best_k}")
print(f"   - MAE optimal : {best_mae:.4f}")
print(f"   - R² optimal : {best_r2:.4f}")

# ============================================
# COMPARAISON AVEC D'AUTRES ALGORITHMES
# ============================================

print("\n" + "=" * 40)
print("COMPARAISON AVEC D'AUTRES ALGORITHMES")
print("=" * 40)

# Dictionnaire des modèles de régression
regression_models = {
    'KNN (optimal)': KNeighborsRegressor(n_neighbors=best_k),
    'Régression Linéaire': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.01),
    'Arbre de décision': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results_regression = []

print("\nEntraînement et évaluation des modèles...")
for name, model in regression_models.items():
    # Entraînement
    model.fit(X_train_reg, y_train_reg)
    
    # Prédictions
    y_train_pred = model.predict(X_train_reg)
    y_test_pred = model.predict(X_test_reg)
    
    # Métriques
    train_mae = mean_absolute_error(y_train_reg, y_train_pred)
    test_mae = mean_absolute_error(y_test_reg, y_test_pred)
    train_r2 = r2_score(y_train_reg, y_train_pred)
    test_r2 = r2_score(y_test_reg, y_test_pred)
    
    results_regression.append({
        'Modèle': name,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R²': train_r2,
        'Test R²': test_r2
    })
    
    print(f"\n{name} :")
    print(f"   Test MAE = {test_mae:.4f}, R² = {test_r2:.4f}")

# Créer un DataFrame pour comparer
results_reg_df = pd.DataFrame(results_regression)
results_reg_df = results_reg_df.sort_values('Test MAE')

print("\n" + "=" * 40)
print("TABLEAU COMPARATIF (trié par Test MAE)")
print("=" * 40)
print(results_reg_df.to_string(index=False))

# Visualisation comparative
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
models = results_reg_df['Modèle']
test_mae_values = results_reg_df['Test MAE']
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
bars = plt.barh(models, test_mae_values, color=colors)
plt.xlabel('MAE (plus petit = mieux)')
plt.title('Comparaison des MAE en test')
for bar, val in zip(bars, test_mae_values):
    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center')

plt.subplot(1, 2, 2)
test_r2_values = results_reg_df['Test R²']
bars = plt.barh(models, test_r2_values, color=colors)
plt.xlabel('R² (plus grand = mieux)')
plt.title('Comparaison des R² en test')
plt.xlim([0, 1])
for bar, val in zip(bars, test_r2_values):
    plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center')

plt.tight_layout()
plt.savefig('regression_models_comparison.png', dpi=150)
plt.show()

# ============================================
# ANALYSE DES RÉSIDUS
# ============================================

print("\n" + "=" * 40)
print("ANALYSE DES RÉSIDUS")
print("=" * 40)

# Prendre le meilleur modèle (celui avec le plus petit MAE)
best_model_name = results_reg_df.iloc[0]['Modèle']
best_model = regression_models[best_model_name]

print(f"Meilleur modèle : {best_model_name}")

# Prédictions avec le meilleur modèle
y_test_pred_best = best_model.predict(X_test_reg)
residuals = y_test_reg - y_test_pred_best

# Graphique des résidus
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus (réel - prédit)')
plt.title(f'Résidus vs Prédictions - {best_model_name}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.title('Distribution des résidus')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=150)
plt.show()

print(f"\nStatistiques des résidus :")
print(f"   - Moyenne : {residuals.mean():.4f} (devrait être proche de 0)")
print(f"   - Écart-type : {residuals.std():.4f}")
print(f"   - Min : {residuals.min():.4f}")
print(f"   - Max : {residuals.max():.4f}")

# ============================================
# VALIDATION CROISÉE POUR LE MEILLEUR MODÈLE
# ============================================

print("\n" + "=" * 40)
print("VALIDATION CROISÉE")
print("=" * 40)

# Validation croisée sur le meilleur modèle
cv_scores_mae = cross_val_score(best_model, X_scaled, y, cv=5, 
                                 scoring='neg_mean_absolute_error')
cv_scores_r2 = cross_val_score(best_model, X_scaled, y, cv=5, 
                                scoring='r2')

print(f"\nValidation croisée 5-fold pour {best_model_name} :")
print(f"   - MAE moyen : {-cv_scores_mae.mean():.4f} ± {cv_scores_mae.std():.4f}")
print(f"   - R² moyen : {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")

# ============================================
# OPTIMISATION POUR LE MODÈLE CHOISI
# ============================================

print("\n" + "=" * 40)
print(f"OPTIMISATION DE {best_model_name}")
print("=" * 40)

if best_model_name == 'Random Forest':
    # GridSearch pour Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search_rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    print("Recherche des meilleurs paramètres...")
    grid_search_rf.fit(X_train_reg, y_train_reg)
    
    print(f"\nMeilleurs paramètres :")
    for param, value in grid_search_rf.best_params_.items():
        print(f"   - {param} : {value}")
    
    best_model_final = grid_search_rf.best_estimator_
    
elif best_model_name == 'Gradient Boosting':
    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0]
    }
    
    grid_search_gb = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid_gb,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    print("Recherche des meilleurs paramètres...")
    grid_search_gb.fit(X_train_reg, y_train_reg)
    
    print(f"\nMeilleurs paramètres :")
    for param, value in grid_search_gb.best_params_.items():
        print(f"   - {param} : {value}")
    
    best_model_final = grid_search_gb.best_estimator_
    
else:
    best_model_final = best_model
    print(f"Pas d'optimisation supplémentaire pour {best_model_name}")

# Évaluation finale
y_test_pred_final = best_model_final.predict(X_test_reg)
final_mae = mean_absolute_error(y_test_reg, y_test_pred_final)
final_mse = mean_squared_error(y_test_reg, y_test_pred_final)
final_r2 = r2_score(y_test_reg, y_test_pred_final)

print(f"\nPerformances finales :")
print(f"   - MAE : {final_mae:.4f}")
print(f"   - MSE : {final_mse:.4f}")
print(f"   - R²  : {final_r2:.4f}")

# ============================================
# VISUALISATION DES PRÉDICTIONS FINALES
# ============================================

plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_test_pred_final, alpha=0.6)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 
         'r--', linewidth=2, label='Prédiction parfaite')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title(f'Prédictions vs Réalité - {best_model_name}\nMAE={final_mae:.2f}, R²={final_r2:.2%}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_predictions.png', dpi=150)
plt.show()

# ============================================
# SAUVEGARDE DU MODÈLE POUR DÉPLOIEMENT
# ============================================

print("\n" + "=" * 40)
print("SAUVEGARDE DU MODÈLE POUR DÉPLOIEMENT")
print("=" * 40)

import pickle

# Sauvegarder le meilleur modèle
with open('auto-mpg.pkl', 'wb') as f:
    pickle.dump(best_model_final, f)

# Sauvegarder aussi le scaler
with open('auto-mpg-scaler.pkl', 'wb') as f:
    pickle.dump(scaler_reg, f)

print(f"✓ Modèle sauvegardé sous 'auto-mpg.pkl'")
print(f"✓ Scaler sauvegardé sous 'auto-mpg-scaler.pkl'")

# ============================================
# CONCLUSION DE LA PARTIE 2
# ============================================

print("\n" + "=" * 60)
print("CONCLUSION DE LA PARTIE 2")
print("=" * 60)

print(f"""
Résumé des résultats :

1. Meilleur modèle : {best_model_name}
   - MAE : {final_mae:.2f} mpg
   - MSE : {final_mse:.2f}
   - R²  : {final_r2:.2%}

2. Interprétation :
   - En moyenne, l'erreur de prédiction est de {final_mae:.2f} mpg
   - Le modèle explique {final_r2:.2%} de la variance de la consommation

3. Comparaison avec KNN optimal :
   - KNN (k={best_k}) : MAE={best_mae:.2f}, R²={best_r2:.2%}
   - Amélioration avec {best_model_name} : {best_mae/final_mae:.2f}x mieux

4. Importance de la normalisation :
   - Cruciale pour KNN et modèles basés sur les distances
   - Moins importante pour les arbres

5. Pour le déploiement :
   - Modèle sauvegardé dans auto-mpg.pkl
   - Prêt à être intégré dans Streamlit
""")

# ============================================
# BONUS : FEATURE IMPORTANCE POUR LA RÉGRESSION
# ============================================

if hasattr(best_model_final, 'feature_importances_'):
    print("\n" + "=" * 40)
    print("FEATURE IMPORTANCE POUR LA RÉGRESSION")
    print("=" * 40)
    
    importances = best_model_final.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title(f'Importance des variables - {best_model_name}')
    plt.tight_layout()
    plt.savefig('regression_feature_importance.png', dpi=150)
    plt.show()


print("\n✅ Analyse terminée ! Tous les graphiques ont été sauvegardés.")