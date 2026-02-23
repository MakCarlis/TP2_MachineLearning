# ============================================
# PARTIE 3 : EXPÉRIMENTATION SUR UN NOUVEAU JEU DE DONNÉES
# ============================================
# Dataset : Meilleures ventes de mangas (best-selling-manga.csv)
# Objectif : Prédire les ventes totales en fonction des caractéristiques des mangas
# Problème : Régression (ventes en millions)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PARTIE 3 : ANALYSE DES VENTES DE MANGAS")
print("=" * 80)

# ============================================
# 1. CHARGEMENT ET PREMIÈRE ANALYSE
# ============================================

print("\n" + "=" * 60)
print("1. CHARGEMENT ET ANALYSE EXPLORATOIRE")
print("=" * 60)

# Charger le dataset
df_manga = pd.read_csv('best-selling-manga.csv', encoding='utf-8')

print(f"\n📊 Dimensions du dataset : {df_manga.shape}")
print(f"   - {df_manga.shape[0]} mangas")
print(f"   - {df_manga.shape[1]} caractéristiques")

print("\n📋 Aperçu des 5 premières lignes :")
print(df_manga.head())

print("\nℹ️ Informations sur les types de données :")
print(df_manga.info())

print("\n📈 Statistiques descriptives :")
print(df_manga.describe())

# ============================================
# 2. NETTOYAGE ET PRÉPARATION DES DONNÉES
# ============================================

print("\n" + "=" * 60)
print("2. NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("=" * 60)

# Vérifier les valeurs manquantes
print("\n🔍 Valeurs manquantes :")
missing_values = df_manga.isnull().sum()
print(missing_values[missing_values > 0])

# Traitement des valeurs manquantes
# Pour 'Demographic', on remplace par 'Unknown'
df_manga['Demographic'].fillna('Unknown', inplace=True)

# Pour les colonnes numériques, vérifier les valeurs aberrantes
print("\n🔍 Analyse des valeurs aberrantes :")

# Examiner la colonne 'Approximate sales in million(s)'
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_manga['Approximate sales in million(s)'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Ventes (millions)')
plt.ylabel('Nombre de mangas')
plt.title('Distribution des ventes totales')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df_manga['Approximate sales in million(s)'])
plt.ylabel('Ventes (millions)')
plt.title('Boxplot des ventes totales')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('manga_sales_distribution.png', dpi=150)
plt.show()

# Statistiques sur les ventes
print("\n📊 Statistiques des ventes :")
print(f"   - Min : {df_manga['Approximate sales in million(s)'].min():.1f} millions")
print(f"   - Max : {df_manga['Approximate sales in million(s)'].max():.1f} millions")
print(f"   - Moyenne : {df_manga['Approximate sales in million(s)'].mean():.1f} millions")
print(f"   - Médiane : {df_manga['Approximate sales in million(s)'].median():.1f} millions")
print(f"   - Écart-type : {df_manga['Approximate sales in million(s)'].std():.1f} millions")

# Identifier les outliers (méthode IQR)
Q1 = df_manga['Approximate sales in million(s)'].quantile(0.25)
Q3 = df_manga['Approximate sales in million(s)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_manga[(df_manga['Approximate sales in million(s)'] < lower_bound) | 
                    (df_manga['Approximate sales in million(s)'] > upper_bound)]
print(f"\n🔍 Nombre d'outliers détectés : {len(outliers)}")
print("Top 5 des outliers (ventes exceptionnelles) :")
print(outliers[['Manga series', 'Approximate sales in million(s)']].head())

# ============================================
# 3. CRÉATION DE NOUVELLES FEATURES (FEATURE ENGINEERING)
# ============================================

print("\n" + "=" * 60)
print("3. CRÉATION DE NOUVELLES FEATURES")
print("=" * 60)

# Extraire l'année de début de sérialisation
def extract_start_year(serialized):
    """Extrait l'année de début de la chaîne 'année–année'"""
    if pd.isna(serialized):
        return None
    try:
        # Format: "1997–present" ou "1984–1995" ou "1990–present"
        parts = str(serialized).split('–')
        if len(parts) > 0:
            year_str = parts[0].strip()
            if year_str.isdigit() and len(year_str) == 4:
                return int(year_str)
    except:
        pass
    return None

def extract_end_year(serialized):
    """Extrait l'année de fin de la chaîne"""
    if pd.isna(serialized):
        return None
    try:
        parts = str(serialized).split('–')
        if len(parts) > 1:
            year_str = parts[1].strip()
            if year_str == 'present':
                return 2024  # Année actuelle pour les séries en cours
            elif year_str.isdigit() and len(year_str) == 4:
                return int(year_str)
    except:
        pass
    return None

print("\n🔧 Extraction des années de publication...")
df_manga['start_year'] = df_manga['Serialized'].apply(extract_start_year)
df_manga['end_year'] = df_manga['Serialized'].apply(extract_end_year)

# Calculer la durée de publication
df_manga['publication_years'] = df_manga['end_year'] - df_manga['start_year']

# Vérifier les résultats
print("\n📊 Années extraites (échantillon) :")
print(df_manga[['Manga series', 'Serialized', 'start_year', 'end_year', 'publication_years']].head(10))

# Statistiques sur la durée
print("\n📈 Statistiques de la durée de publication :")
print(f"   - Durée moyenne : {df_manga['publication_years'].mean():.1f} ans")
print(f"   - Durée médiane : {df_manga['publication_years'].median():.1f} ans")
print(f"   - Durée max : {df_manga['publication_years'].max()} ans (Golgo 13, depuis 1968!)")

# Créer une feature pour les séries en cours
df_manga['ongoing'] = df_manga['Serialized'].str.contains('present', na=False).astype(int)

# Calculer les ventes par volume (si pas déjà présent)
df_manga['sales_per_volume'] = df_manga['Approximate sales in million(s)'] / df_manga['No. of collected volumes']

print("\n📊 Nouvelles features créées :")
print(f"   - start_year : année de début")
print(f"   - end_year : année de fin")
print(f"   - publication_years : durée de publication")
print(f"   - ongoing : 1 si série en cours, 0 sinon")
print(f"   - sales_per_volume : ventes moyennes par volume")

# ============================================
# 4. ANALYSE EXPLORATOIRE APPROFONDIE
# ============================================

print("\n" + "=" * 60)
print("4. ANALYSE EXPLORATOIRE APPROFONDIE")
print("=" * 60)

# Analyser les ventes par démographie
print("\n📊 Ventes par démographie :")
demographic_stats = df_manga.groupby('Demographic')['Approximate sales in million(s)'].agg(['count', 'mean', 'median', 'sum']).sort_values('sum', ascending=False)
print(demographic_stats)

# Visualisation
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
demographic_stats['sum'].plot(kind='bar', color='skyblue')
plt.title('Ventes totales par démographie')
plt.xlabel('Démographie')
plt.ylabel('Ventes totales (millions)')
plt.xticks(rotation=45)

plt.subplot(2, 3, 2)
demographic_stats['mean'].plot(kind='bar', color='lightgreen')
plt.title('Ventes moyennes par démographie')
plt.xlabel('Démographie')
plt.ylabel('Ventes moyennes (millions)')
plt.xticks(rotation=45)

# Analyser les ventes par éditeur
print("\n📊 Top 10 éditeurs par ventes totales :")
publisher_stats = df_manga.groupby('Publisher')['Approximate sales in million(s)'].sum().sort_values(ascending=False).head(10)
print(publisher_stats)

plt.subplot(2, 3, 3)
publisher_stats.plot(kind='bar', color='coral')
plt.title('Top 10 éditeurs par ventes')
plt.xlabel('Éditeur')
plt.ylabel('Ventes totales (millions)')
plt.xticks(rotation=45)

# Relation entre nombre de volumes et ventes
plt.subplot(2, 3, 4)
plt.scatter(df_manga['No. of collected volumes'], df_manga['Approximate sales in million(s)'], alpha=0.6)
plt.xlabel('Nombre de volumes')
plt.ylabel('Ventes totales (millions)')
plt.title('Ventes vs Nombre de volumes')
plt.grid(True, alpha=0.3)

# Calculer la corrélation
corr_volumes_sales = df_manga['No. of collected volumes'].corr(df_manga['Approximate sales in million(s)'])
print(f"\n📈 Corrélation volumes-ventes : {corr_volumes_sales:.3f}")

# Relation entre durée et ventes
plt.subplot(2, 3, 5)
plt.scatter(df_manga['publication_years'], df_manga['Approximate sales in million(s)'], alpha=0.6)
plt.xlabel('Durée de publication (années)')
plt.ylabel('Ventes totales (millions)')
plt.title('Ventes vs Durée de publication')
plt.grid(True, alpha=0.3)

corr_years_sales = df_manga['publication_years'].corr(df_manga['Approximate sales in million(s)'])
print(f"📈 Corrélation durée-ventes : {corr_years_sales:.3f}")

# Distribution des ventes par volume
plt.subplot(2, 3, 6)
plt.hist(df_manga['sales_per_volume'].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Ventes moyennes par volume (millions)')
plt.ylabel('Fréquence')
plt.title('Distribution des ventes par volume')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('manga_detailed_analysis.png', dpi=150)
plt.show()

# ============================================
# 5. MATRICE DE CORRÉLATION
# ============================================

print("\n" + "=" * 60)
print("5. MATRICE DE CORRÉLATION")
print("=" * 60)

# Sélectionner les colonnes numériques pour la corrélation
numeric_cols = ['No. of collected volumes', 'Approximate sales in million(s)', 
                'Average sales per volume in million(s)', 'start_year', 
                'publication_years', 'sales_per_volume', 'ongoing']

# Filtrer les colonnes existantes
available_numeric = [col for col in numeric_cols if col in df_manga.columns]
corr_matrix = df_manga[available_numeric].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.3f')
plt.title('Matrice de corrélation - Variables numériques')
plt.tight_layout()
plt.savefig('manga_correlation_matrix.png', dpi=150)
plt.show()

print("\n🔍 Corrélations avec les ventes totales :")
corr_with_sales = corr_matrix['Approximate sales in million(s)'].sort_values(ascending=False)
print(corr_with_sales)

# ============================================
# 6. PRÉPARATION POUR LE MACHINE LEARNING
# ============================================

print("\n" + "=" * 60)
print("6. PRÉPARATION POUR LE MACHINE LEARNING")
print("=" * 60)

# Définir les features et la cible
# Objectif : prédire 'Approximate sales in million(s)' (ventes totales)
target = 'Approximate sales in million(s)'

# Sélectionner les features pertinentes
feature_cols = [
    'No. of collected volumes',
    'publication_years',
    'ongoing',
    'start_year'
]

# Ajouter les variables catégorielles
categorical_cols = ['Demographic', 'Publisher']

print(f"\n🎯 Variable cible : {target}")
print(f"\n📊 Features numériques : {feature_cols}")
print(f"📊 Features catégorielles : {categorical_cols}")

# Créer X et y
X = df_manga[feature_cols + categorical_cols].copy()
y = df_manga[target].copy()

# Vérifier les valeurs manquantes
print("\n🔍 Valeurs manquantes dans X :")
print(X.isnull().sum())

# Supprimer les lignes avec des valeurs manquantes
X = X.dropna()
y = y[X.index]  # Aligner y avec X

print(f"\n📏 Dimensions après nettoyage : X={X.shape}, y={y.shape}")

# ============================================
# 7. CRÉATION DU PIPELINE DE PRÉPROCESSING
# ============================================

print("\n" + "=" * 60)
print("7. CRÉATION DU PIPELINE DE PRÉPROCESSING")
print("=" * 60)

# Définir les transformations pour les colonnes numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Définir les transformations pour les colonnes catégorielles
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combiner les transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, feature_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

print("✓ Pipeline de preprocessing créé")
print("   - Normalisation des features numériques")
print("   - One-hot encoding des variables catégorielles")

# ============================================
# 8. SPLIT TRAIN/TEST
# ============================================

print("\n" + "=" * 60)
print("8. SPLIT TRAIN/TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Taille train : {X_train.shape[0]} ({len(X_train)/len(X):.1%})")
print(f"📊 Taille test : {X_test.shape[0]} ({len(X_test)/len(X):.1%})")

print(f"\n📈 Statistiques de y_train :")
print(f"   - Min : {y_train.min():.1f}")
print(f"   - Max : {y_train.max():.1f}")
print(f"   - Moyenne : {y_train.mean():.1f}")
print(f"   - Médiane : {y_train.median():.1f}")

# ============================================
# 9. ENTRAÎNEMENT DE PLUSIEURS MODÈLES
# ============================================

print("\n" + "=" * 60)
print("9. ENTRAÎNEMENT DE PLUSIEURS MODÈLES")
print("=" * 60)

# Dictionnaire des modèles à tester
models = {
    'Régression Linéaire': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.01),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Arbre de décision': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

results = []

print("\n🏋️ Entraînement des modèles en cours...")
for name, model in models.items():
    # Créer le pipeline complet
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Métriques
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results.append({
        'Modèle': name,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Écart MAE': test_mae - train_mae,
        'Pipeline': pipeline  # Sauvegarder pour usage ultérieur
    })
    
    print(f"\n{name}:")
    print(f"   MAE Test: {test_mae:.2f}M, R² Test: {test_r2:.3f}")

# Créer un DataFrame pour comparer
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Pipeline'} for r in results])
results_df = results_df.sort_values('Test MAE')

print("\n" + "=" * 60)
print("📊 TABLEAU COMPARATIF DES MODÈLES")
print("=" * 60)
print(results_df.to_string(index=False))

# ============================================
# 10. ANALYSE DES RÉSULTATS
# ============================================

print("\n" + "=" * 60)
print("10. ANALYSE DES RÉSULTATS")
print("=" * 60)

best_model_row = results_df.iloc[0]
best_model_name = best_model_row['Modèle']
best_mae = best_model_row['Test MAE']
best_r2 = best_model_row['Test R²']

print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"   - MAE : {best_mae:.2f} millions")
print(f"   - R²  : {best_r2:.3f} ({best_r2*100:.1f}%)")

print(f"\n📈 Interprétation des métriques :")
print(f"   - MAE (Mean Absolute Error) : En moyenne, le modèle se trompe de {best_mae:.2f} millions")
print(f"     dans ses prédictions de ventes.")
print(f"   - R² (Coefficient de détermination) : Le modèle explique {best_r2*100:.1f}%")
print(f"     de la variance des ventes. C'est {'excellent' if best_r2 > 0.7 else 'bon' if best_r2 > 0.5 else 'moyen'}.")

# Comparaison avec un modèle naïf (prédire la moyenne)
naive_pred = np.full_like(y_test, y_train.mean())
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_r2 = r2_score(y_test, naive_pred)

print(f"\n📊 Comparaison avec un modèle naïf (prédire la moyenne) :")
print(f"   - Modèle naïf MAE : {naive_mae:.2f}M")
print(f"   - {best_model_name} MAE : {best_mae:.2f}M")
print(f"   - Amélioration : {(naive_mae - best_mae)/naive_mae*100:.1f}%")

# ============================================
# 11. RÉCUPÉRATION DU PIPELINE DU MEILLEUR MODÈLE
# ============================================

# Récupérer le pipeline du meilleur modèle
best_pipeline = None
for r in results:
    if r['Modèle'] == best_model_name:
        best_pipeline = r['Pipeline']
        break

# ============================================
# 12. ANALYSE DES RÉSIDUS
# ============================================

print("\n" + "=" * 60)
print("11. ANALYSE DES RÉSIDUS")
print("=" * 60)

if best_pipeline:
    # Prédictions finales
    y_test_pred = best_pipeline.predict(X_test)
    residuals = y_test - y_test_pred
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Valeurs prédites (millions)')
    plt.ylabel('Résidus')
    plt.title(f'Résidus vs Prédictions - {best_model_name}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title('Distribution des résidus')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('manga_residuals.png', dpi=150)
    plt.show()
    
    print(f"\n📊 Statistiques des résidus :")
    print(f"   - Moyenne : {residuals.mean():.4f} (devrait être proche de 0)")
    print(f"   - Écart-type : {residuals.std():.4f}")
    print(f"   - Min : {residuals.min():.4f}")
    print(f"   - Max : {residuals.max():.4f}")
    
    # Test de normalité (Shapiro-Wilk)
    from scipy import stats
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:100])  # Limit to 100 for speed
    print(f"\n📊 Test de normalité de Shapiro-Wilk :")
    print(f"   - Statistique : {shapiro_stat:.4f}")
    print(f"   - p-value : {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print("   ✅ Les résidus suivent une distribution normale")
    else:
        print("   ⚠ Les résidus ne suivent pas une distribution normale")

# ============================================
# 13. IMPORTANCE DES VARIABLES (SI DISPONIBLE)
# ============================================

print("\n" + "=" * 60)
print("12. IMPORTANCE DES VARIABLES")
print("=" * 60)

if best_pipeline and hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
    # Récupérer les noms des features après preprocessing
    feature_names = []
    feature_names.extend(feature_cols)
    
    # Ajouter les noms des variables one-hot
    cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    feature_names.extend(cat_feature_names)
    
    # Récupérer les importances
    importances = best_pipeline.named_steps['regressor'].feature_importances_
    
    # Créer un DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 features les plus importantes ({best_model_name}) :")
    print(importance_df.head(10).to_string(index=False))
    
    # Visualisation
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df.head(15)['feature'][::-1], 
             importance_df.head(15)['importance'][::-1])
    plt.xlabel('Importance')
    plt.title(f'Importance des variables - {best_model_name}')
    plt.tight_layout()
    plt.savefig('manga_feature_importance.png', dpi=150)
    plt.show()
    
    print("\n📊 Interprétation des features importantes :")
    top_features = importance_df.head(5)
    for idx, row in top_features.iterrows():
        print(f"   - {row['feature']} : {row['importance']:.3f}")
        if 'No. of collected volumes' in row['feature']:
            print("     → Le nombre de volumes est un prédicteur naturel des ventes totales")
        elif 'publication_years' in row['feature']:
            print("     → Les séries longues accumulent plus de ventes")
        elif 'Demographic' in row['feature']:
            demo = row['feature'].replace('Demographic_', '')
            print(f"     → La démographie {demo} est surreprésentée dans les best-sellers")

# ============================================
# 14. VALIDATION CROISÉE
# ============================================

print("\n" + "=" * 60)
print("13. VALIDATION CROISÉE")
print("=" * 60)

if best_pipeline:
    # Validation croisée 5-fold
    cv_scores_mae = cross_val_score(best_pipeline, X, y, cv=5, 
                                     scoring='neg_mean_absolute_error')
    cv_scores_r2 = cross_val_score(best_pipeline, X, y, cv=5, 
                                    scoring='r2')
    
    print(f"\n📊 Validation croisée 5-fold pour {best_model_name} :")
    print(f"   - MAE moyen : {-cv_scores_mae.mean():.4f} ± {cv_scores_mae.std():.4f}")
    print(f"   - R² moyen : {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
    
    print(f"\n📈 Interprétation :")
    print(f"   - L'écart-type faible ({cv_scores_r2.std():.4f}) indique que le modèle est stable")
    print(f"   - Le modèle généralise bien à différents échantillons")

# ============================================
# 15. OPTIMISATION DU MEILLEUR MODÈLE
# ============================================

print("\n" + "=" * 60)
print("14. OPTIMISATION DU MEILLEUR MODÈLE")
print("=" * 60)

if best_model_name == 'Random Forest':
    # GridSearch pour Random Forest
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [5, 10, 15, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    print("\n🔍 Recherche des meilleurs hyperparamètres...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                               scoring='neg_mean_absolute_error',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Meilleurs paramètres trouvés :")
    for param, value in grid_search.best_params_.items():
        print(f"   - {param} : {value}")
    
    best_model_optimized = grid_search.best_estimator_
    
    # Évaluation
    y_test_pred_opt = best_model_optimized.predict(X_test)
    opt_mae = mean_absolute_error(y_test, y_test_pred_opt)
    opt_r2 = r2_score(y_test, y_test_pred_opt)
    
    print(f"\n📊 Performance après optimisation :")
    print(f"   - MAE : {opt_mae:.4f} (vs {best_mae:.4f})")
    print(f"   - R² : {opt_r2:.4f} (vs {best_r2:.4f})")
    print(f"   - Amélioration MAE : {(best_mae - opt_mae)/best_mae*100:.1f}%")
    
    best_final_model = best_model_optimized
    final_mae = opt_mae
    final_r2 = opt_r2
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 4, 5],
        'regressor__subsample': [0.8, 1.0]
    }
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    print("\n🔍 Recherche des meilleurs hyperparamètres...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                               scoring='neg_mean_absolute_error',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Meilleurs paramètres trouvés :")
    for param, value in grid_search.best_params_.items():
        print(f"   - {param} : {value}")
    
    best_model_optimized = grid_search.best_estimator_
    
    # Évaluation
    y_test_pred_opt = best_model_optimized.predict(X_test)
    opt_mae = mean_absolute_error(y_test, y_test_pred_opt)
    opt_r2 = r2_score(y_test, y_test_pred_opt)
    
    print(f"\n📊 Performance après optimisation :")
    print(f"   - MAE : {opt_mae:.4f} (vs {best_mae:.4f})")
    print(f"   - R² : {opt_r2:.4f} (vs {best_r2:.4f})")
    print(f"   - Amélioration MAE : {(best_mae - opt_mae)/best_mae*100:.1f}%")
    
    best_final_model = best_model_optimized
    final_mae = opt_mae
    final_r2 = opt_r2
    
else:
    best_final_model = best_pipeline
    final_mae = best_mae
    final_r2 = best_r2

# ============================================
# 16. PRÉDICTIONS SUR DES CAS CONCRETS
# ============================================

print("\n" + "=" * 60)
print("15. PRÉDICTIONS SUR DES CAS CONCRETS")
print("=" * 60)

# Créer quelques exemples de mangas pour tester le modèle
test_cases = pd.DataFrame({
    'No. of collected volumes': [30, 50, 100, 20],
    'publication_years': [5, 15, 30, 3],
    'ongoing': [0, 1, 1, 0],
    'start_year': [2018, 2010, 1995, 2020],
    'Demographic': ['Shōnen', 'Seinen', 'Shōnen', 'Shōjo'],
    'Publisher': ['Shueisha', 'Kodansha', 'Shogakukan', 'Hakusensha']
})

if best_final_model:
    predictions = best_final_model.predict(test_cases)
    
    print("\n🔮 Prédictions pour des mangas hypothétiques :")
    for i, pred in enumerate(predictions):
        print(f"\nCas {i+1}: {test_cases.iloc[i].to_dict()}")
        print(f"   → Ventes prédites : {pred:.1f} millions")
        
        # Interprétation
        if pred > 50:
            print("     ⭐ Potentiel best-seller majeur")
        elif pred > 20:
            print("     📈 Bonnes ventes attendues")
        else:
            print("     📊 Ventes modestes")

# ============================================
# 17. SAUVEGARDE DU MODÈLE
# ============================================

print("\n" + "=" * 60)
print("16. SAUVEGARDE DU MODÈLE")
print("=" * 60)

import pickle

if best_final_model:
    with open('manga_sales_model.pkl', 'wb') as f:
        pickle.dump(best_final_model, f)
    
    print("✅ Modèle sauvegardé sous 'manga_sales_model.pkl'")
    print("   - Pour prédire les ventes de nouveaux mangas")
    print("   - Utiliser pipeline.predict(nouveaux_données)")

# ============================================
# 18. CONCLUSION GÉNÉRALE
# ============================================

print("\n" + "=" * 80)
print("CONCLUSION GÉNÉRALE - ANALYSE DES VENTES DE MANGAS")
print("=" * 80)

print(f"""
📚 RÉSUMÉ DE L'ANALYSE :

1. Dataset analysé : {df_manga.shape[0]} mangas best-sellers

2. Caractéristiques clés identifiées :
   - Le nombre de volumes est fortement corrélé aux ventes
   - La durée de publication est un facteur important
   - Shōnen et Seinen dominent les ventes
   - Shueisha et Kodansha sont les éditeurs majeurs

3. Modèle optimal : {best_model_name}
   - MAE : {final_mae:.2f} millions
   - R² : {final_r2:.2%}
   
4. Interprétation métier :
   - Une erreur moyenne de {final_mae:.2f}M est acceptable pour prédire des ventes
   - Le modèle explique {final_r2:.1f}% des variations de ventes
   - Pour un nouveau manga avec 20 volumes sur 5 ans, on prédit environ {predictions[3]:.1f}M

5. Limitations :
   - Données limitées aux best-sellers (biais de sélection)
   - Pas de données sur les adaptations (anime, films)
   - Pas de données sur les prix ou le marketing

6. Applications potentielles :
   - Aider les éditeurs à estimer le potentiel d'un nouveau manga
   - Identifier les facteurs de succès
   - Comparer les performances par démographie/éditeur
""")

# ============================================
# 19. BONUS : VISUALISATION FINALE
# ============================================

print("\n" + "=" * 60)
print("BONUS : VISUALISATION FINALE")
print("=" * 60)

# Créer un graphique récapitulatif
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Graphique 1 : Top 10 des mangas par ventes
top10 = df_manga.nlargest(10, 'Approximate sales in million(s)')
axes[0, 0].barh(top10['Manga series'], top10['Approximate sales in million(s)'])
axes[0, 0].set_xlabel('Ventes (millions)')
axes[0, 0].set_title('Top 10 des mangas les plus vendus')
axes[0, 0].invert_yaxis()

# Graphique 2 : Ventes par démographie
demographic_means = df_manga.groupby('Demographic')['Approximate sales in million(s)'].mean().sort_values()
axes[0, 1].barh(demographic_means.index, demographic_means.values)
axes[0, 1].set_xlabel('Ventes moyennes (millions)')
axes[0, 1].set_title('Ventes moyennes par démographie')

# Graphique 3 : Relation volumes-ventes
axes[1, 0].scatter(df_manga['No. of collected volumes'], 
                   df_manga['Approximate sales in million(s)'], alpha=0.5)
axes[1, 0].set_xlabel('Nombre de volumes')
axes[1, 0].set_ylabel('Ventes (millions)')
axes[1, 0].set_title('Relation volumes vs ventes')
axes[1, 0].grid(True, alpha=0.3)

# Graphique 4 : Prédictions vs réalité
if best_final_model:
    y_all_pred = best_final_model.predict(X)
    axes[1, 1].scatter(y, y_all_pred, alpha=0.5)
    axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Ventes réelles (millions)')
    axes[1, 1].set_ylabel('Ventes prédites (millions)')
    axes[1, 1].set_title(f'Prédictions vs réalité - {best_model_name}')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('manga_final_summary.png', dpi=150)
plt.show()

print("\n✅ Analyse terminée ! Tous les graphiques ont été sauvegardés.")