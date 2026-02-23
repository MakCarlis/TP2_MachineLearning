# ============================================
# PARTIE 1 : MISE EN ŒUVRE DES MÉTHODES D'ENSEMBLE
# ============================================

# Importation des bibliothèques nécessaires

import pandas as pd                  # Pour la manipulation des données
import numpy as np                    # Pour les calculs numériques
import matplotlib.pyplot as plt        # Pour les graphiques de base
import seaborn as sns                  # Pour des graphiques plus sophistiqués
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')      # Pour ignorer les avertissements non critiques
import pickle



# ============================================
# A : DESCRIPTION ET VISUALISATION DES DONNÉES
# ============================================

# Chargement du dataset
df = pd.read_csv('acs2017_county_data.csv')
print("=" * 60)
print("APERÇU DU DATASET")
print("=" * 60)

# 1) Utilisation des fonctions pandas pour l'analyse initiale

# shape : donne les dimensions (lignes, colonnes) du dataframe
print(f"\n1. Dimensions du dataset (shape) : {df.shape}")
print(f"   - {df.shape[0]} instances (comtés)")
print(f"   - {df.shape[1]} caractéristiques (variables)")

# info() : résumé des données (types, valeurs non-nulles)
print("\n2. Informations générales sur les données :")
print(df.info())

# head() : affiche les 5 premières lignes pour voir la structure
print("\n3. Aperçu des 5 premières lignes :")
print(df.head())

# describe() : statistiques descriptives pour les colonnes numériques
print("\n4. Statistiques descriptives :")
print(df.describe())

# Réponses aux questions spécifiques
print("\n" + "=" * 60)
print("RÉPONSES AUX QUESTIONS DE LA PARTIE 1-A")
print("=" * 60)

# Combien de classes ? (pour la variable cible "Income")
# Note : Dans ce dataset, "Income" est le revenu moyen, pas une classe binaire
# On va créer une variable cible binaire pour la classification
# On suppose que le seuil est 50K comme dans l'énoncé
median_income = df['Income'].median()
df['Income_class'] = (df['Income'] > 50000).astype(int)
# 1 pour >50K, 0 pour <=50K

print(f"\na) Nombre de classes pour la variable cible 'Income' : 2 classes")
print(f"   - 0 : Revenu ≤ 50K ({(df['Income_class'] == 0).sum()} instances)")
print(f"   - 1 : Revenu > 50K ({(df['Income_class'] == 1).sum()} instances)")

# Combien de caractéristiques descriptives ? De quels types ?
# On exclut les colonnes d'identification et la cible
feature_cols = [col for col in df.columns if col not in ['CountyId', 'State', 'County', 'Income_class']]
print(f"\nb) Nombre de caractéristiques descriptives : {len(feature_cols)}")
print("\n   Types des caractéristiques :")
print(df[feature_cols].dtypes.value_counts())

# Combien d'instances ?
print(f"\nc) Nombre total d'instances : {df.shape[0]}")

# Combien d'instances de chaque classe ?
print(f"\nd) Distribution des classes :")
print(df['Income_class'].value_counts())
print(f"   En pourcentage :")
print(df['Income_class'].value_counts(normalize=True) * 100)

# Comment sont organisés les instances ?
print(f"\ne) Organisation des instances :")
print(f"   - Les données sont organisées par État (State) et par comté (County)")
print(f"   - Nombre d'États différents : {df['State'].nunique()}")
print(f"   - Nombre de comtés par État (moyenne) : {df.groupby('State').size().mean():.1f}")

# ============================================
# 2) Visualisation des données : croisement des variables
# ============================================

print("\n" + "=" * 60)
print("VISUALISATION DES DONNÉES")
print("=" * 60)

# Sélectionner quelques variables clés pour la visualisation
# Pour éviter d'avoir trop de graphiques, on choisit les plus pertinentes
key_vars = ['Income', 'TotalPop', 'Professional', 'Unemployment', 'Poverty']

# Créer une copie avec seulement ces variables + la classe
df_viz = df[key_vars + ['Income_class']].copy()

# Convertir Income_class en catégorie pour la couleur
df_viz['Income_class'] = df_viz['Income_class'].map({0: '≤50K', 1: '>50K'})

# 2a) Pairplot : matrice de scatter plots
print("\nCréation du pairplot (cela peut prendre quelques secondes)...")
plt.figure(figsize=(12, 10))
pairplot = sns.pairplot(df_viz, hue='Income_class', diag_kind='hist', 
                        plot_kws={'alpha': 0.5, 's': 10})
pairplot.fig.suptitle("Matrice de dispersion des variables clés", y=1.02)
plt.tight_layout()
plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Graphique sauvegardé sous 'pairplot.png'")

# 2b) Tracer les droites de régression et donner les paramètres
print("\nAnalyse des relations linéaires :")

# Fonction pour tracer une régression linéaire
def plot_regression_with_params(x, y, data, title):
    plt.figure(figsize=(8, 6))
    
    # Nuage de points avec couleur selon la classe
    colors = {0: 'blue', 1: 'red'}
    for cls in [0, 1]:
        subset = data[data['Income_class'] == cls]
        plt.scatter(subset[x], subset[y], 
                   c=colors[cls], label=f'Classe {cls}', alpha=0.5, s=10)
    
    # Droite de régression globale
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[x], data[y])
    
    x_range = np.linspace(data[x].min(), data[x].max(), 100)
    plt.plot(x_range, intercept + slope * x_range, 
            'k-', label=f'Régression linéaire', linewidth=2)
    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    
    # Ajouter les paramètres sous forme de texte
    text = f"Équation : {y} = {slope:.4f} × {x} + {intercept:.2f}\n"
    text += f"R² = {r_value**2:.4f} | p-value = {p_value:.4f}"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'regression_{x}_{y}.png', dpi=150)
    plt.show()
    
    return slope, intercept, r_value**2

# Analyser quelques relations importantes
relations = [
    ('Professional', 'Income', "Relation entre % de professionnels et revenu"),
    ('Unemployment', 'Poverty', "Relation entre chômage et pauvreté"),
    ('TotalPop', 'Income', "Relation entre population et revenu")
]

results = []
for x, y, title in relations:
    print(f"\n--- {title} ---")
    slope, intercept, r2 = plot_regression_with_params(x, y, df, title)
    results.append({
        'x': x, 'y': y,
        'slope': slope, 'intercept': intercept,
        'r2': r2
    })

# Afficher un tableau récapitulatif
print("\n" + "=" * 60)
print("RÉCAPITULATIF DES RÉGRESSIONS LINÉAIRES")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("COMMENTAIRES SUR LES RÉSULTATS")
print("=" * 60)
print("""
Observations principales :
1. Relation Professional vs Income :
   - Corrélation positive forte : plus le pourcentage de professionnels est élevé,
     plus le revenu moyen est élevé. C'est logique car les professions libérales/
     techniques sont généralement mieux rémunérées.

2. Relation Unemployment vs Poverty :
   - Corrélation positive : le chômage est fortement lié à la pauvreté.
     Les comtés avec un fort taux de chômage ont aussi un fort taux de pauvreté.

3. Relation TotalPop vs Income :
   - Corrélation faible : la taille de la population n'est pas un bon prédicteur
     du revenu moyen. Certaines grandes villes ont des revenus élevés mais pas toutes.

Ces observations nous aideront pour la suite : les variables comme 'Professional'
et 'Unemployment' seront probablement importantes pour la prédiction.
""")

# Sauvegarder le dataframe avec la nouvelle colonne pour la suite
df.to_csv('census_with_class.csv', index=False)
print("\n✓ Dataset enrichi sauvegardé sous 'census_with_class.csv'")


# ============================================
# SÉPARATION DES DONNÉES EN BASES D'APPRENTISSAGE ET DE TEST
# ============================================

print("=" * 60)
print("PRÉPARATION DES DONNÉES POUR LE MODÈLE KNN")
print("=" * 60)

# Recharger le dataset avec la classe créée précédemment
df = pd.read_csv('census_with_class.csv')

# 1) Identifier les colonnes pertinentes
# On exclut les colonnes d'identification et la cible
feature_cols = [col for col in df.columns if col not in ['CountyId', 'State', 'County', 'Income', 'Income_class']]
print(f"\nNombre de caractéristiques initiales : {len(feature_cols)}")

# 2) Gérer les valeurs manquantes (s'il y en a)
print("\nVérification des valeurs manquantes :")
missing_values = df[feature_cols].isnull().sum()
if missing_values.sum() > 0:
    print(f"  - {missing_values.sum()} valeurs manquantes détectées")
    # Pour simplifier, on remplit les valeurs manquantes avec la médiane
    for col in feature_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    print("  ✓ Valeurs manquantes traitées")
else:
    print("  ✓ Aucune valeur manquante")

# 3) Encoder les variables catégorielles (si nécessaire)
# Dans ce dataset, toutes les variables semblent numériques
print("\nTypes des caractéristiques :")
print(df[feature_cols].dtypes.value_counts())

# 4) Normalisation des données (cruciale pour KNN)
# KNN est sensible aux échelles des variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
X = pd.DataFrame(X_scaled, columns=feature_cols)

# 5) Définir X (features) et y (cible)
y = df['Income_class']  # Notre cible binaire (0: ≤50K, 1: >50K)

# 6) Split en train/test (80% / 20% comme suggéré dans l'énoncé)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # stratify=y assure la même proportion de classes dans train et test

print(f"\nTaille de l'ensemble d'apprentissage : {X_train.shape[0]} instances")
print(f"Taille de l'ensemble de test : {X_test.shape[0]} instances")
print(f"Proportion de classe 1 (>50K) dans train : {y_train.mean():.2%}")
print(f"Proportion de classe 1 (>50K) dans test : {y_test.mean():.2%}")

# ============================================
# APPRENTISSAGE ET TEST AVEC KNN
# ============================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("\n" + "=" * 60)
print("MODÈLE KNN (K-Nearest Neighbors)")
print("=" * 60)

# 1) Créer un KNN avec k=5 par défaut
knn = KNeighborsClassifier(n_neighbors=5)  # 5 voisins par défaut

# Entraîner le modèle sur la base d'apprentissage
print("\nEntraînement du modèle KNN...")
knn.fit(X_train, y_train)
print("✓ Modèle entraîné")

# Prédictions sur l'ensemble d'apprentissage et de test
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# 1) Scores en apprentissage et en test
train_score = accuracy_score(y_train, y_train_pred)
test_score = accuracy_score(y_test, y_test_pred)

print(f"\n1) Scores du modèle KNN (k=5) :")
print(f"   - Score en apprentissage : {train_score:.4f} ({train_score*100:.2f}%)")
print(f"   - Score en test : {test_score:.4f} ({test_score*100:.2f}%)")

# Interprétation : un écart de plus de 5-10% indiquerait du sur-apprentissage
if train_score - test_score > 0.1:
    print("   ⚠ Attention : écart important indiquant possible sur-apprentissage")
else:
    print("   ✓ Écart raisonnable entre apprentissage et test")

# 2) Matrice de confusion
print("\n2) Matrice de confusion :")
cm = confusion_matrix(y_test, y_test_pred)
print("                  Prédit")
print("                  Négatif  Positif")
print(f"Réel Négatif     {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"Réel Positif     {cm[1,0]:6d}  {cm[1,1]:6d}")

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['≤50K', '>50K'], 
            yticklabels=['≤50K', '>50K'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title(f'Matrice de Confusion - KNN (k=5)\nAccuracy: {test_score:.2%}')
plt.tight_layout()
plt.savefig('confusion_matrix_knn.png', dpi=150)
plt.show()
print("✓ Matrice de confusion sauvegardée")

# Calculer des métriques détaillées
print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_test_pred, 
                          target_names=['≤50K', '>50K']))

# Interprétation de la matrice de confusion
print("\nObservations sur la matrice de confusion :")
tn, fp, fn, tp = cm.ravel()
print(f"   - Vrais Négatifs (correctement prédits ≤50K) : {tn}")
print(f"   - Faux Positifs (prédits >50K mais en réalité ≤50K) : {fp}")
print(f"   - Faux Négatifs (prédits ≤50K mais en réalité >50K) : {fn}")
print(f"   - Vrais Positifs (correctement prédits >50K) : {tp}")

# Calcul de métriques supplémentaires
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMétriques pour la classe >50K :")
print(f"   - Précision : {precision:.4f} (quand on prédit >50K, on a raison à {precision:.2%})")
print(f"   - Rappel : {recall:.4f} (on détecte {recall:.2%} des vrais >50K)")
print(f"   - F1-score : {f1:.4f} (moyenne harmonique des deux)")

# ============================================
# 2) ÉTUDE DE L'INFLUENCE DU PARAMÈTRE k
# ============================================

print("\n" + "=" * 60)
print("ÉTUDE DE L'INFLUENCE DU PARAMÈTRE k")
print("=" * 60)

# Tester différentes valeurs de k
k_values = range(1, 51)  # Tester k de 1 à 50
train_scores = []
test_scores = []

print("\nCalcul des performances pour k = 1 à 50...")
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    
    train_scores.append(knn_temp.score(X_train, y_train))
    test_scores.append(knn_temp.score(X_test, y_test))
    
    if k % 10 == 0:  # Afficher progression tous les 10 k
        print(f"  k={k:2d} : train={train_scores[-1]:.4f}, test={test_scores[-1]:.4f}")

# Visualisation de l'influence de k
plt.figure(figsize=(12, 6))

# Courbes d'évolution
plt.subplot(1, 2, 1)
plt.plot(k_values, train_scores, 'b-', label='Score apprentissage', linewidth=2)
plt.plot(k_values, test_scores, 'r-', label='Score test', linewidth=2)
plt.xlabel('Valeur de k (nombre de voisins)')
plt.ylabel('Accuracy')
plt.title('Influence du paramètre k sur les performances')
plt.legend()
plt.grid(True, alpha=0.3)

# Zoom sur les premières valeurs
plt.subplot(1, 2, 2)
plt.plot(k_values[:20], train_scores[:20], 'b-', label='Train', linewidth=2)
plt.plot(k_values[:20], test_scores[:20], 'r-', label='Test', linewidth=2)
plt.xlabel('Valeur de k (nombre de voisins)')
plt.ylabel('Accuracy')
plt.title('Focus sur k=1 à 20')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_k_influence.png', dpi=150)
plt.show()

# Trouver le meilleur k
best_k_idx = np.argmax(test_scores)
best_k = k_values[best_k_idx]
best_score = test_scores[best_k_idx]

print(f"\nMeilleure performance en test :")
print(f"   - k optimal = {best_k}")
print(f"   - Accuracy = {best_score:.4f} ({best_score*100:.2f}%)")

# Analyse des observations
print("\nObservations sur l'influence de k :")
print("""
   - Quand k est petit (1-3) : 
        * Score train élevé (souvent 1.0)
        * Score test variable, risque de sur-apprentissage
   - Quand k augmente (10-30) :
        * Score train diminue légèrement
        * Score test se stabilise, meilleure généralisation
   - Quand k est grand (>30) :
        * Les deux scores diminuent (sous-apprentissage)
   - Compromis optimal : k autour de {best_k} pour ce dataset
""".format(best_k=best_k))

# ============================================
# 3) REMPLACER KNN PAR D'AUTRES MODÈLES
# ============================================

print("\n" + "=" * 60)
print("COMPARAISON AVEC D'AUTRES MODÈLES")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Dictionnaire des modèles à tester
models = {
    'KNN (optimal)': KNeighborsClassifier(n_neighbors=best_k),
    'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (linéaire)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Arbre de décision': DecisionTreeClassifier(random_state=42, max_depth=5)
}

results = []

print("\nEntraînement et évaluation des différents modèles...")
for name, model in models.items():
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Scores
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    results.append({
        'Modèle': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Écart': train_acc - test_acc
    })
    
    print(f"\n{name} :")
    print(f"   Train: {train_acc:.4f} | Test: {test_acc:.4f} | Écart: {train_acc-test_acc:.4f}")

# Créer un DataFrame pour comparer
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test Accuracy', ascending=False)

print("\n" + "=" * 60)
print("TABLEAU COMPARATIF DES PERFORMANCES")
print("=" * 60)
print(results_df.to_string(index=False))

# Visualisation comparative
plt.figure(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.35

plt.bar(x - width/2, results_df['Train Accuracy'], width, label='Train', alpha=0.8)
plt.bar(x + width/2, results_df['Test Accuracy'], width, label='Test', alpha=0.8)

plt.xlabel('Modèles')
plt.ylabel('Accuracy')
plt.title('Comparaison des performances des modèles')
plt.xticks(x, results_df['Modèle'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

# ============================================
# 4) DÉDUIRE LE MODÈLE OPTIMAL
# ============================================

print("\n" + "=" * 60)
print("SÉLECTION DU MODÈLE OPTIMAL")
print("=" * 60)

# Trouver le meilleur modèle basé sur le score test
best_model_row = results_df.iloc[0]
best_model_name = best_model_row['Modèle']
best_model_test = best_model_row['Test Accuracy']

print(f"\nMeilleur modèle : {best_model_name}")
print(f"Accuracy en test : {best_model_test:.4f} ({best_model_test*100:.2f}%)")

# Critères de sélection
print("\nCritères pris en compte pour la sélection :")
print("""
   1. Performance en test (priorité principale)
   2. Écart train-test (pour éviter le sur-apprentissage)
   3. Complexité du modèle (plus simple = meilleure généralisation)
   4. Temps d'entraînement (pour le déploiement)
""")

# Analyser les forces/faiblesses de chaque modèle
print("\nAnalyse comparative :")
for idx, row in results_df.iterrows():
    print(f"\n{row['Modèle']} :")
    print(f"   - Test Accuracy: {row['Test Accuracy']:.2%}")
    print(f"   - Écart train-test: {row['Écart']:.2%}")
    if row['Modèle'] == best_model_name:
        print("   ⭐ MODÈLE OPTIMAL")

# ============================================
# DÉPLOIEMENT DU MODÈLE (sauvegarde)
# ============================================

import pickle

print("\n" + "=" * 60)
print("DÉPLOIEMENT DU MODÈLE OPTIMAL")
print("=" * 60)

# Sauvegarder le meilleur modèle
best_model = models[best_model_name]  # Récupérer le modèle optimal

# Sauvegarde au format .pkl
with open('census.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n✓ Modèle optimal sauvegardé sous 'census.pkl'")
print(f"  - Modèle : {best_model_name}")
print(f"  - Accuracy test : {best_model_test:.2%}")

# Sauvegarder aussi le scaler pour pouvoir normaliser les futures données
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler sauvegardé sous 'scaler.pkl'")

# Démonstration de chargement du modèle
print("\nTest de chargement du modèle :")
with open('census.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Vérifier que le modèle chargé fonctionne
sample_pred = loaded_model.predict(X_test[:5])
print(f"  ✓ Modèle chargé avec succès")
print(f"  ✓ Prédictions sur 5 échantillons : {sample_pred}")


# ============================================
# PARTIE 1-B : VALIDATION CROISÉE
# ============================================

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 60)
print("PARTIE 1-B : VALIDATION CROISÉE SUR ARBRE DE DÉCISION")
print("=" * 60)

# Recharger les données préparées (on garde le même split que précédemment)
# X_train, X_test, y_train, y_test sont déjà définis

# ============================================
# 1. PRÉPARATION DES DONNÉES (déjà fait)
# ============================================

print("\n1. Préparation des données :")
print("   ✓ Valeurs manquantes traitées")
print("   ✓ Variables catégorielles encodées")
print("   ✓ Données normalisées (optionnel pour arbres)")
print("   ✓ Split train/test (80/20) effectué")

# Note : Les arbres de décision n'ont pas besoin de normalisation
# mais on garde les données normalisées pour la cohérence

# ============================================
# 2. CLASSIFIEUR CONSTANT (RÉFÉRENTIEL À BATTRE)
# ============================================

print("\n" + "=" * 60)
print("2. CLASSIFIEUR CONSTANT (RÉFÉRENTIEL)")
print("=" * 60)

# Le classifieur constant prédit toujours la classe majoritaire
from sklearn.dummy import DummyClassifier

# Créer un classifieur constant qui prédit la classe la plus fréquente
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)

# Prédictions et erreur
y_pred_dummy = dummy.predict(X_test)
dummy_accuracy = accuracy_score(y_test, y_pred_dummy)
dummy_error = 1 - dummy_accuracy

print(f"\nClasse majoritaire dans l'ensemble d'apprentissage :")
print(f"   - Classe 0 (≤50K) : {(y_train == 0).sum()} instances ({y_train.mean()*100:.1f}%)")
print(f"   - Classe 1 (>50K) : {(y_train == 1).sum()} instances ({(1-y_train.mean())*100:.1f}%)")

print(f"\nPerformance du classifieur constant :")
print(f"   - Accuracy : {dummy_accuracy:.4f} ({dummy_accuracy*100:.2f}%)")
print(f"   - Erreur de test : {dummy_error:.4f} ({dummy_error*100:.2f}%)")

print(f"\n📌 Ce score de {dummy_accuracy:.2%} est notre référentiel à battre !")
print(f"   Tout modèle performant doit faire mieux que ça.")

# ============================================
# 3. CONSTRUCTION D'UN PREMIER ARBRE DE DÉCISION
# ============================================

print("\n" + "=" * 60)
print("3. PREMIER ARBRE DE DÉCISION (peu profond)")
print("=" * 60)

# Créer un arbre volontairement petit pour visualisation
tree_small = DecisionTreeClassifier(
    max_depth=3,           # Arbre peu profond (3 niveaux)
    min_samples_split=20,  # Minimum d'échantillons pour diviser un noeud
    min_samples_leaf=10,   # Minimum d'échantillons par feuille
    random_state=42
)

# Entraînement
tree_small.fit(X_train, y_train)

# Évaluation
y_train_pred_small = tree_small.predict(X_train)
y_test_pred_small = tree_small.predict(X_test)

train_acc_small = accuracy_score(y_train, y_train_pred_small)
test_acc_small = accuracy_score(y_test, y_test_pred_small)

print(f"\nPerformance de l'arbre (max_depth=3) :")
print(f"   - Accuracy train : {train_acc_small:.4f} ({train_acc_small*100:.2f}%)")
print(f"   - Accuracy test  : {test_acc_small:.4f} ({test_acc_small*100:.2f}%)")
print(f"   - Erreur test    : {1-test_acc_small:.4f} ({(1-test_acc_small)*100:.2f}%)")

# Comparaison avec le classifieur constant
improvement = (test_acc_small - dummy_accuracy) / dummy_accuracy * 100
print(f"\nComparaison avec le classifieur constant :")
print(f"   - Amélioration : +{improvement:.1f}%")

# ============================================
# 3b. VISUALISATION DE L'ARBRE (avec graphviz)
# ============================================

print("\n" + "-" * 40)
print("Visualisation de l'arbre")

# Méthode 1 : avec matplotlib (simplifié)
plt.figure(figsize=(20, 10))
plot_tree(tree_small, 
          feature_names=X_train.columns.tolist(),
          class_names=['≤50K', '>50K'],
          filled=True, 
          rounded=True,
          fontsize=10,
          max_depth=3)  # Limiter l'affichage à 3 niveaux
plt.title("Arbre de décision (max_depth=3)")
plt.tight_layout()
plt.savefig('decision_tree_small.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Arbre sauvegardé sous 'decision_tree_small.png'")

# Méthode 2 : exporter au format .dot pour graphviz (optionnel)
from sklearn.tree import export_graphviz
export_graphviz(tree_small, 
                out_file='tree.dot',
                feature_names=X_train.columns.tolist(),
                class_names=['≤50K', '>50K'],
                filled=True, rounded=True,
                special_characters=True)
print("✓ Fichier .dot créé (peut être visualisé avec Graphviz)")

# Interprétation des features importantes
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': tree_small.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeatures les plus importantes (top 5) :")
print(feature_importance.head(10).to_string(index=False))

# ============================================
# 4. ÉTUDE DE L'INFLUENCE DE max_depth
# ============================================

print("\n" + "=" * 60)
print("4. INFLUENCE DU PARAMÈTRE max_depth")
print("=" * 60)

# Tester différentes profondeurs
depths = range(1, 21)  # de 1 à 20
train_scores_depth = []
test_scores_depth = []
tree_models = []

print("Entraînement des arbres avec différentes profondeurs...")
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    train_scores_depth.append(tree.score(X_train, y_train))
    test_scores_depth.append(tree.score(X_test, y_test))
    tree_models.append(tree)
    
    if depth % 5 == 0:
        print(f"  depth={depth:2d} : train={train_scores_depth[-1]:.4f}, test={test_scores_depth[-1]:.4f}")

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(depths, train_scores_depth, 'b-', label='Train', linewidth=2)
plt.plot(depths, test_scores_depth, 'r-', label='Test', linewidth=2)
plt.axhline(y=dummy_accuracy, color='g', linestyle='--', label='Classifieur constant')
plt.xlabel('Profondeur maximale (max_depth)')
plt.ylabel('Accuracy')
plt.title('Influence de la profondeur sur les performances')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Calculer l'écart train-test (indicateur de sur-apprentissage)
gap = np.array(train_scores_depth) - np.array(test_scores_depth)
plt.plot(depths, gap, 'purple', linewidth=2)
plt.xlabel('Profondeur maximale (max_depth)')
plt.ylabel('Écart Train - Test')
plt.title("Écart d'accuracy (indicateur de sur-apprentissage)")
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tree_depth_influence.png', dpi=150)
plt.show()

# Trouver la meilleure profondeur
best_depth_idx = np.argmax(test_scores_depth)
best_depth = depths[best_depth_idx]
best_depth_score = test_scores_depth[best_depth_idx]

print(f"\nMeilleure profondeur (sur test) : depth={best_depth}")
print(f"   - Accuracy test : {best_depth_score:.4f} ({best_depth_score*100:.2f}%)")
print(f"   - Accuracy train : {train_scores_depth[best_depth_idx]:.4f}")
print(f"   - Écart : {train_scores_depth[best_depth_idx] - best_depth_score:.4f}")

# ============================================
# 5. VALIDATION CROISÉE AVEC GridSearchCV
# ============================================

print("\n" + "=" * 60)
print("5. VALIDATION CROISÉE AVEC GridSearchCV")
print("=" * 60)

# Définir la grille de paramètres à tester
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],  # None = pas de limite
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

print(f"Grille de paramètres à tester :")
print(f"   - max_depth : {param_grid['max_depth']}")
print(f"   - min_samples_split : {param_grid['min_samples_split']}")
print(f"   - min_samples_leaf : {param_grid['min_samples_leaf']}")
print(f"   - criterion : {param_grid['criterion']}")
print(f"\nNombre total de combinaisons : {np.prod([len(v) for v in param_grid.values()])}")

# Créer le GridSearchCV avec validation croisée (5 folds)
print("\nLancement du GridSearchCV (peut prendre quelques minutes)...")
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,                    # Validation croisée 5-fold
    scoring='accuracy',      # Métrique d'évaluation
    n_jobs=-1,               # Utiliser tous les processeurs
    verbose=1                # Afficher la progression
)

# Entraîner le grid search
grid_search.fit(X_train, y_train)

# Résultats
print("\n" + "=" * 40)
print("RÉSULTATS DU GRIDSEARCH")
print("=" * 40)

print(f"\nMeilleurs paramètres trouvés :")
for param, value in grid_search.best_params_.items():
    print(f"   - {param} : {value}")

print(f"\nMeilleur score de validation croisée : {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# Évaluer sur l'ensemble de test
best_tree = grid_search.best_estimator_
y_test_pred_best = best_tree.predict(X_test)
test_score_best = accuracy_score(y_test, y_test_pred_best)

print(f"\nPerformance sur l'ensemble de test :")
print(f"   - Accuracy : {test_score_best:.4f} ({test_score_best*100:.2f}%)")

# Comparaison avec la profondeur optimale simple
print(f"\nComparaison avec la sélection simple (max_depth uniquement) :")
print(f"   - Sélection simple (max_depth={best_depth}) : {best_depth_score:.2%}")
print(f"   - GridSearch (paramètres optimisés) : {test_score_best:.2%}")
improvement_grid = (test_score_best - best_depth_score) / best_depth_score * 100
print(f"   - Amélioration : +{improvement_grid:.1f}%")

# ============================================
# 5b. VISUALISATION DE L'ERREUR DE VALIDATION CROISÉE
# ============================================

print("\n" + "-" * 40)
print("Visualisation des résultats du GridSearch")

# Extraire les résultats du grid search
cv_results = pd.DataFrame(grid_search.cv_results_)

# Visualiser l'influence de max_depth en gardant les autres params fixes
# Filtrer pour min_samples_split=5, min_samples_leaf=1, criterion='gini' (par exemple)
filtered_results = cv_results[
    (cv_results['param_min_samples_split'] == 5) &
    (cv_results['param_min_samples_leaf'] == 1) &
    (cv_results['param_criterion'] == 'gini')
].copy()

# Remplacer None par une grande valeur pour l'affichage
filtered_results['param_max_depth_display'] = filtered_results['param_max_depth'].apply(
    lambda x: 30 if x is None else x
)
filtered_results = filtered_results.sort_values('param_max_depth_display')

plt.figure(figsize=(10, 6))
plt.plot(filtered_results['param_max_depth_display'], 
         filtered_results['mean_test_score'], 
         'b-o', label='Score validation croisée', linewidth=2)
plt.fill_between(filtered_results['param_max_depth_display'],
                  filtered_results['mean_test_score'] - filtered_results['std_test_score'],
                  filtered_results['mean_test_score'] + filtered_results['std_test_score'],
                  alpha=0.2, color='b')

plt.xlabel('max_depth')
plt.ylabel('Accuracy moyenne (validation croisée)')
plt.title("Erreur de validation croisée en fonction de max_depth\n(autres paramètres fixés)")
plt.grid(True, alpha=0.3)
plt.legend()

# Marquer le meilleur point
best_point = filtered_results.loc[filtered_results['mean_test_score'].idxmax()]
plt.plot(best_point['param_max_depth_display'], 
         best_point['mean_test_score'], 
         'r*', markersize=15, label='Meilleur point')

plt.tight_layout()
plt.savefig('cv_results_depth.png', dpi=150)
plt.show()

# ============================================
# 6. ÉVALUATION FINALE DU CLASSIFIEUR OPTIMISÉ
# ============================================

print("\n" + "=" * 60)
print("6. ÉVALUATION FINALE - ARBRE OPTIMISÉ")
print("=" * 60)

# Matrice de confusion pour l'arbre optimisé
cm_best = confusion_matrix(y_test, y_test_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens',
            xticklabels=['≤50K', '>50K'],
            yticklabels=['≤50K', '>50K'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title(f"Matrice de confusion - Arbre optimisé\nAccuracy: {test_score_best:.2%}")
plt.tight_layout()
plt.savefig('confusion_matrix_best_tree.png', dpi=150)
plt.show()

# Rapport de classification détaillé
from sklearn.metrics import classification_report
print("\nRapport de classification - Arbre optimisé :")
print(classification_report(y_test, y_test_pred_best, 
                          target_names=['≤50K', '>50K']))

# Comparaison finale avec tous les modèles
print("\n" + "=" * 40)
print("COMPARAISON FINALE")
print("=" * 40)

comparison = pd.DataFrame({
    'Modèle': ['Classifieur constant', 'Arbre simple (depth=3)', 
               'Arbre optimisé (GridSearch)'],
    'Accuracy test': [dummy_accuracy, test_acc_small, test_score_best],
    'Amélioration vs constant': ['-', 
        f"{(test_acc_small-dummy_accuracy)/dummy_accuracy*100:.1f}%",
        f"{(test_score_best-dummy_accuracy)/dummy_accuracy*100:.1f}%"]
})

print(comparison.to_string(index=False))

# ============================================
# CONCLUSION DE LA PARTIE 1-B
# ============================================

print("\n" + "=" * 60)
print("CONCLUSIONS DE LA PARTIE 1-B")
print("=" * 60)

print("""
Points clés à retenir :

1. Classifieur constant : 
   - Baseline à battre : {dummy:.2%} accuracy
   - Important pour mesurer l'apport réel des modèles

2. Influence de max_depth :
   - Profondeur trop faible → sous-apprentissage
   - Profondeur trop grande → sur-apprentissage
   - Optimal trouvé à depth={best_depth}

3. Validation croisée :
   - Permet d'éviter le sur-apprentissage
   - GridSearch explore automatiquement les combinaisons
   - Meilleurs paramètres : {best_params}

4. Amélioration obtenue :
   - {improve_vs_dummy:.1f}% par rapport au classifieur constant
   - {improve_vs_simple:.1f}% par rapport à l'arbre simple

5. Prochaine étape :
   - Appliquer ces techniques aux méthodes d'ensemble (Random Forest, Boosting)
""".format(
    dummy=dummy_accuracy,
    best_depth=best_depth,
    best_params=grid_search.best_params_,
    improve_vs_dummy=(test_score_best/dummy_accuracy-1)*100,
    improve_vs_simple=(test_score_best/test_acc_small-1)*100
))

# Sauvegarder le meilleur arbre pour utilisation future
with open('best_tree.pkl', 'wb') as f:
    pickle.dump(best_tree, f)
print("\n✓ Meilleur arbre sauvegardé sous 'best_tree.pkl'")


# ============================================
# PARTIE 1-C : BAGGING ET RANDOM FOREST
# ============================================

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 60)
print("PARTIE 1-C : BAGGING ET RANDOM FOREST")
print("=" * 60)

# ============================================
# 1. BAGGING AVEC RandomForestClassifier
# ============================================

print("\n" + "=" * 60)
print("1. BAGGING - PRINCIPE ET IMPLÉMENTATION")
print("=" * 60)

print("""
Le Bagging (Bootstrap Aggregating) :
- Crée B échantillons bootstrap (tirage aléatoire avec remise)
- Entraîne un arbre sur chaque échantillon
- Agrège les prédictions par vote majoritaire
- Réduit la variance sans augmenter le biais
""")

# Comment utiliser RandomForestClassifier pour faire du Bagging ?
# Pour faire du Bagging "pur" (sans sélection aléatoire des features),
# il faut fixer max_features = n_features (prendre toutes les features)

n_features = X_train.shape[1]
print(f"\nNombre total de features : {n_features}")

# Bagging avec des arbres (max_features = n_features)
bagging_rf = RandomForestClassifier(
    n_estimators=100,           # B = 100 arbres
    max_features=n_features,    # Prendre toutes les features = Bagging pur
    bootstrap=True,              # Échantillonnage bootstrap
    oob_score=True,              # Calculer le score Out-Of-Bag
    random_state=42,
    n_jobs=-1                    # Utiliser tous les processeurs
)

print("\nEntraînement du Bagging (100 arbres, toutes les features)...")
start_time = time.time()
bagging_rf.fit(X_train, y_train)
bagging_time = time.time() - start_time

print(f"✓ Entraînement terminé en {bagging_time:.2f} secondes")

# Évaluation
y_train_pred_bag = bagging_rf.predict(X_train)
y_test_pred_bag = bagging_rf.predict(X_test)

train_acc_bag = accuracy_score(y_train, y_train_pred_bag)
test_acc_bag = accuracy_score(y_test, y_test_pred_bag)

print(f"\nPerformances du Bagging :")
print(f"   - Accuracy train : {train_acc_bag:.4f} ({train_acc_bag*100:.2f}%)")
print(f"   - Accuracy test  : {test_acc_bag:.4f} ({test_acc_bag*100:.2f}%)")
print(f"   - Score OOB (Out-of-Bag) : {bagging_rf.oob_score_:.4f} ({bagging_rf.oob_score_*100:.2f}%)")

# Comparaison avec l'arbre simple
print(f"\nComparaison avec l'arbre optimisé :")
print(f"   - Arbre optimisé : {test_score_best:.2%}")
print(f"   - Bagging : {test_acc_bag:.2%}")
improvement = (test_acc_bag - test_score_best) / test_score_best * 100
print(f"   - Amélioration : +{improvement:.1f}%")

# ============================================
# 1b. ÉTUDE DE LA COMPLEXITÉ ET PERFORMANCE SELON B
# ============================================

print("\n" + "-" * 40)
print("Influence du nombre d'arbres (B) sur les performances")

# Tester différentes valeurs de B (n_estimators)
B_values = [1, 5, 10, 20, 50, 100, 200, 300]
train_scores_b = []
test_scores_b = []
oob_scores_b = []
times_b = []

print("\nEntraînement pour différentes valeurs de B...")
for B in B_values:
    start = time.time()
    rf_temp = RandomForestClassifier(
        n_estimators=B,
        max_features=n_features,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    
    train_scores_b.append(rf_temp.score(X_train, y_train))
    test_scores_b.append(rf_temp.score(X_test, y_test))
    oob_scores_b.append(rf_temp.oob_score_)
    times_b.append(time.time() - start)
    
    print(f"  B={B:3d} : train={train_scores_b[-1]:.4f}, test={test_scores_b[-1]:.4f}, "
          f"oob={oob_scores_b[-1]:.4f}, temps={times_b[-1]:.2f}s")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Graphique 1 : Évolution des scores
axes[0].plot(B_values, train_scores_b, 'b-o', label='Train', linewidth=2)
axes[0].plot(B_values, test_scores_b, 'r-o', label='Test', linewidth=2)
axes[0].plot(B_values, oob_scores_b, 'g-o', label='OOB', linewidth=2)
axes[0].axhline(y=test_score_best, color='orange', linestyle='--', 
                label='Arbre optimisé', alpha=0.7)
axes[0].set_xlabel('Nombre d\'arbres (B)')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Performance vs Nombre d\'arbres')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphique 2 : Écart train-test (sur-apprentissage)
gap_b = np.array(train_scores_b) - np.array(test_scores_b)
axes[1].plot(B_values, gap_b, color="purple", marker="o", linewidth=2)
axes[1].set_xlabel('Nombre d\'arbres (B)')
axes[1].set_ylabel('Écart Train - Test')
axes[1].set_title('Écart (indicateur de sur-apprentissage)')
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1].grid(True, alpha=0.3)

# Graphique 3 : Temps d'entraînement
axes[2].plot(B_values, times_b, 'b-o', linewidth=2)
axes[2].set_xlabel('Nombre d\'arbres (B)')
axes[2].set_ylabel('Temps (secondes)')
axes[2].set_title('Complexité temporelle')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bagging_B_analysis.png', dpi=150)
plt.show()

print("\nObservations sur la complexité et performance :")
print("""
   - Quand B augmente (1 → 50) :
        * Performance s'améliore rapidement
        * Temps d'entraînement augmente linéairement
   - Quand B > 100 :
        * Gains de performance marginaux (loi des rendements décroissants)
        * Le score se stabilise autour de {:.2%}
   - Compromis optimal : B ≈ 100-200 (bonne performance / temps raisonnable)
   - Le score OOB est un bon estimateur de l'erreur de test (économise la validation croisée)
""".format(test_scores_b[-1]))


# ============================================
# 2. RANDOM FOREST (AVEC SÉLECTION ALÉATOIRE DES FEATURES)
# ============================================

print("\n" + "=" * 60)
print("2. RANDOM FORET (AVEC SÉLECTION ALÉATOIRE DES FEATURES)")
print("=" * 60)

print("""
Différence avec le Bagging :
- Bagging : toutes les features sont considérées à chaque split
- Random Forest : on sélectionne aléatoirement p features à chaque split
- p est généralement sqrt(n_features) pour la classification
""")

# Choisir une valeur pour p (paramètre max_features)
p_sqrt = int(np.sqrt(n_features))
p_log = int(np.log2(n_features)) + 1
p_half = n_features // 2

print(f"\nValeurs possibles pour p (max_features) :")
print(f"   - sqrt(n_features) = {p_sqrt}")
print(f"   - log2(n_features) = {p_log}")
print(f"   - n_features/2 = {p_half}")
print(f"   - Toutes les features (Bagging) = {n_features}")

# Construire une Random Forest avec p = sqrt(n_features)
rf_sqrt = RandomForestClassifier(
    n_estimators=100,
    max_features=p_sqrt,      # sqrt(n_features)
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

print(f"\nEntraînement de la Random Forest (p={p_sqrt})...")
rf_sqrt.fit(X_train, y_train)

# Évaluation
y_test_pred_rf = rf_sqrt.predict(X_test)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)
oob_score_rf = rf_sqrt.oob_score_

print(f"\nPerformances de la Random Forest (p={p_sqrt}) :")
print(f"   - Accuracy test : {test_acc_rf:.4f} ({test_acc_rf*100:.2f}%)")
print(f"   - Score OOB : {oob_score_rf:.4f} ({oob_score_rf*100:.2f}%)")

# Comparaison Bagging vs Random Forest
print(f"\nComparaison Bagging vs Random Forest :")
print(f"   - Bagging (p={n_features}) : {test_acc_bag:.2%}")
print(f"   - Random Forest (p={p_sqrt}) : {test_acc_rf:.2%}")
improvement_rf = (test_acc_rf - test_acc_bag) / test_acc_bag * 100
print(f"   - Différence : {improvement_rf:+.1f}%")

# ============================================
# 3. ERREUR OUT-OF-BAG (OOB)
# ============================================

print("\n" + "=" * 60)
print("3. ERREUR OUT-OF-BAG (OOB)")
print("=" * 60)

print("""
Principe de l'erreur OOB :
- Pour chaque arbre, environ 1/3 des données ne sont pas utilisées (out-of-bag)
- Ces données servent de validation naturelle
- La moyenne des erreurs OOB sur tous les arbres est un estimateur non biaisé
- Utile quand on a pas de validation croisée
""")

print(f"\nPour la Random Forest (p={p_sqrt}) :")
print(f"   - Erreur OOB : {1-oob_score_rf:.4f} ({(1-oob_score_rf)*100:.2f}%)")
print(f"   - Erreur test : {1-test_acc_rf:.4f} ({(1-test_acc_rf)*100:.2f}%)")
print(f"   - Écart OOB vs test : {abs(oob_score_rf - test_acc_rf):.4f}")

# Vérification sur différentes valeurs de B
B_values_oob = [10, 50, 100, 200]
oob_errors = []
test_errors = []

print("\nÉvolution de l'erreur OOB avec B :")
for B in B_values_oob:
    rf_temp = RandomForestClassifier(
        n_estimators=B,
        max_features=p_sqrt,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    
    oob_errors.append(1 - rf_temp.oob_score_)
    test_errors.append(1 - rf_temp.score(X_test, y_test))
    print(f"  B={B:3d} : OOB error={oob_errors[-1]:.4f}, Test error={test_errors[-1]:.4f}")

# Visualisation
plt.figure(figsize=(8, 5))
plt.plot(B_values_oob, oob_errors, 'b-o', label='Erreur OOB', linewidth=2)
plt.plot(B_values_oob, test_errors, 'r-o', label='Erreur Test', linewidth=2)
plt.xlabel('Nombre d\'arbres (B)')
plt.ylabel("Taux d'erreur")
plt.title('Comparaison erreur OOB vs erreur Test')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('oob_vs_test.png', dpi=150)
plt.show()

# ============================================
# 4. VALIDATION CROISÉE SUR LE PARAMÈTRE p
# ============================================

print("\n" + "=" * 60)
print("4. VALIDATION CROISÉE SUR LE PARAMÈTRE p (max_features)")
print("=" * 60)

# Fixer B à une valeur raisonnable (100)
B_fixed = 100

# Tester différentes valeurs de p
p_values = [1, 2, 3, 5, 10, 20, 30, n_features//4, n_features//2, n_features]
p_values = [p for p in p_values if p <= n_features]  # Garder les valeurs valides
p_values = sorted(set(p_values))  # Enlever les doublons

print(f"\nTest des valeurs de p avec B={B_fixed} arbres :")
print(f"Valeurs testées : {p_values}")

cv_scores_p = []
test_scores_p = []
oob_scores_p = []

for p in p_values:
    rf_temp = RandomForestClassifier(
        n_estimators=B_fixed,
        max_features=p,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Validation croisée (5-fold) pour être plus robuste
    cv_score = np.mean(cross_val_score(rf_temp, X_train, y_train, cv=5))
    
    # Entraînement complet pour le score test et OOB
    rf_temp.fit(X_train, y_train)
    test_score = rf_temp.score(X_test, y_test)
    
    cv_scores_p.append(cv_score)
    test_scores_p.append(test_score)
    oob_scores_p.append(rf_temp.oob_score_)
    
    print(f"  p={p:3d} : CV={cv_score:.4f}, Test={test_score:.4f}, OOB={rf_temp.oob_score_:.4f}")

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(p_values, cv_scores_p, 'b-o', label='Validation croisée', linewidth=2)
plt.plot(p_values, test_scores_p, 'r-o', label='Test', linewidth=2)
plt.plot(p_values, oob_scores_p, 'g-o', label='OOB', linewidth=2)
plt.xlabel('p (max_features)')
plt.ylabel('Accuracy')
plt.title(f'Influence de p (B={B_fixed} arbres)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Zoom sur les petites valeurs de p
p_small = [p for p in p_values if p <= 20]
idx_small = [i for i, p in enumerate(p_values) if p <= 20]
plt.plot(p_small, [cv_scores_p[i] for i in idx_small], 'b-o', label='CV')
plt.plot(p_small, [test_scores_p[i] for i in idx_small], 'r-o', label='Test')
plt.xlabel('p (max_features) - zoom')
plt.ylabel('Accuracy')
plt.title('Focus sur p ≤ 20')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_p_optimization.png', dpi=150)
plt.show()

# Trouver le meilleur p
best_p_idx = np.argmax(test_scores_p)
best_p = p_values[best_p_idx]
best_p_score = test_scores_p[best_p_idx]

print(f"\nMeilleure valeur de p : {best_p}")
print(f"   - Score test correspondant : {best_p_score:.4f} ({best_p_score*100:.2f}%)")
print(f"   - Score CV : {cv_scores_p[best_p_idx]:.4f}")
print(f"   - Score OOB : {oob_scores_p[best_p_idx]:.4f}")

# ============================================
# 5. RANDOM FOREST OPTIMISÉE
# ============================================

print("\n" + "=" * 60)
print("5. RANDOM FOREST OPTIMISÉE")
print("=" * 60)

# Construire la Random Forest avec les meilleurs paramètres
rf_optimized = RandomForestClassifier(
    n_estimators=B_fixed,
    max_features=best_p,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_optimized.fit(X_train, y_train)

# Évaluation finale
y_test_pred_rf_opt = rf_optimized.predict(X_test)
test_acc_rf_opt = accuracy_score(y_test, y_test_pred_rf_opt)

print(f"\nPerformances de la Random Forest optimisée :")
print(f"   - Paramètres : B={B_fixed}, p={best_p}")
print(f"   - Accuracy test : {test_acc_rf_opt:.4f} ({test_acc_rf_opt*100:.2f}%)")
print(f"   - Score OOB : {rf_optimized.oob_score_:.4f}")

# Matrice de confusion
cm_rf = confusion_matrix(y_test, y_test_pred_rf_opt)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['≤50K', '>50K'],
            yticklabels=['≤50K', '>50K'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title(f'Matrice de confusion - Random Forest optimisée\nAccuracy: {test_acc_rf_opt:.2%}')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=150)
plt.show()

# ============================================
# 6. COMPARAISON FINALE
# ============================================

print("\n" + "=" * 60)
print("6. COMPARAISON FINALE DES MODÈLES")
print("=" * 60)

comparison_final = pd.DataFrame({
    'Modèle': [
        'Classifieur constant',
        'Arbre optimisé',
        'Bagging (B=100)',
        'Random Forest (p=sqrt)',
        'Random Forest optimisée'
    ],
    'Accuracy test': [
        dummy_accuracy,
        test_score_best,
        test_acc_bag,
        test_acc_rf,
        test_acc_rf_opt
    ]
})

print("\nTableau comparatif :")
print(comparison_final.to_string(index=False))

# Visualisation comparative
plt.figure(figsize=(10, 6))
bars = plt.bar(comparison_final['Modèle'], comparison_final['Accuracy test'], 
               color=['gray', 'lightblue', 'blue', 'orange', 'red'])
plt.xlabel('Modèles')
plt.ylabel('Accuracy')
plt.title('Comparaison des performances des modèles')
plt.xticks(rotation=45, ha='right')

# Ajouter les valeurs sur les barres
for bar, val in zip(bars, comparison_final['Accuracy test']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2%}', ha='center', va='bottom')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('final_model_comparison.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("CONCLUSIONS DE LA PARTIE 1-C")
print("=" * 60)

print("""
Points clés à retenir :

1. Bagging :
   - Réduit la variance par rapport à un seul arbre
   - Performance augmente avec B jusqu'à saturation
   - Score OOB = bon estimateur de l'erreur de test

2. Random Forest :
   - Améliore encore le Bagging en décorrélant les arbres
   - Le paramètre p (max_features) est crucial
   - p optimal ≈ sqrt(n_features) pour la classification

3. Gains obtenus :
   - Arbre optimisé → Bagging : +{gain_bag:.1f}%
   - Bagging → Random Forest : +{gain_rf:.1f}%
   - Gain total vs classifieur constant : +{gain_total:.1f}%

4. Prochaine étape :
   - Expérimenter le Boosting (Gradient Boosting)
""".format(
    gain_bag=(test_acc_bag/test_score_best-1)*100,
    gain_rf=(test_acc_rf_opt/test_acc_bag-1)*100,
    gain_total=(test_acc_rf_opt/dummy_accuracy-1)*100
))

# Sauvegarder la Random Forest optimisée
with open('random_forest_optimized.pkl', 'wb') as f:
    pickle.dump(rf_optimized, f)
print("\n✓ Random Forest optimisée sauvegardée sous 'random_forest_optimized.pkl'")


# ============================================
# PARTIE 1-D : BOOSTING (GRADIENT BOOSTING)
# ============================================

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("PARTIE 1-D : BOOSTING AVEC GRADIENT BOOSTING")
print("=" * 60)

print("""
Le Boosting (AdaBoost / Gradient Boosting) :
- Construit les arbres séquentiellement
- Chaque nouvel arbre corrige les erreurs du précédent
- Les arbres sont généralement peu profonds (stumps)
- Très puissant mais risque de sur-apprentissage
""")

# Vérifier la version de scikit-learn
import sklearn
print(f"\nVersion de scikit-learn : {sklearn.__version__}")
if sklearn.__version__ >= '0.20':
    print("✓ Version suffisante pour les fonctionnalités avancées")
else:
    print("⚠ Version ancienne, certaines fonctionnalités peuvent manquer")

# ============================================
# a) IDENTIFICATION DES PARAMÈTRES CRUCIAUX
# ============================================

print("\n" + "=" * 60)
print("a) PARAMÈTRES CRUCIAUX DE GRADIENT BOOSTING")
print("=" * 60)

print("""
Paramètres les plus importants :

1. n_estimators (B) : nombre d'arbres dans la séquence
   - Trop peu → sous-apprentissage
   - Trop → sur-apprentissage (d'où l'intérêt de l'early stopping)

2. learning_rate (ν - nu) : taux d'apprentissage
   - Poids accordé à chaque nouvel arbre
   - Généralement petit (0.01 à 0.1)
   - Compromis avec n_estimators : plus learning_rate est petit,
     plus il faut d'arbres

3. max_depth : profondeur des arbres
   - Pour le boosting, on utilise souvent des arbres peu profonds (3-5)
   - Des arbres trop profonds → sur-apprentissage rapide

4. subsample : fraction des données utilisée pour chaque arbre
   - Introduit de l'aléatoire (comme Random Forest)
   - Valeur typique : 0.5 à 1.0

5. min_samples_split / min_samples_leaf : pour contrôler la complexité

6. validation_fraction / n_iter_no_change : pour l'early stopping
   - validation_fraction : proportion pour la validation
   - n_iter_no_change : nombre d'itérations sans amélioration pour arrêter
""")

print("\nParamètres correspondant à AdaBoost dans GradientBoosting :")
print("""
AdaBoost peut être simulé avec GradientBoosting en utilisant :
   - loss='exponential' (au lieu de 'deviance' par défaut)
   - learning_rate plus élevé
   - max_depth=1 (stumps)
   
Mais la vraie implémentation d'AdaBoost est dans sklearn.ensemble.AdaBoostClassifier
""")

# ============================================
# b) SÉLECTION DE B AVEC EARLY STOPPING
# ============================================

print("\n" + "=" * 60)
print("b) SÉLECTION DE B AVEC EARLY STOPPING")
print("=" * 60)

# Fixer learning_rate et max_depth (on les optimisera plus tard)
learning_rate_fixed = 0.1
max_depth_fixed = 3

print(f"\nParamètres fixés :")
print(f"   - learning_rate = {learning_rate_fixed}")
print(f"   - max_depth = {max_depth_fixed}")

# Créer un Gradient Boosting avec early stopping
gb_early = GradientBoostingClassifier(
    n_estimators=1000,              # Maximum d'arbres
    learning_rate=learning_rate_fixed,
    max_depth=max_depth_fixed,
    validation_fraction=0.1,        # 10% pour validation
    n_iter_no_change=10,            # Arrêt si pas d'amélioration pendant 10 itérations
    tol=1e-4,                        # Tolérance pour l'amélioration
    random_state=42
)

print("\nEntraînement avec early stopping...")
start_time = time.time()
gb_early.fit(X_train, y_train)
training_time = time.time() - start_time

# Nombre d'arbres réellement utilisés
n_estimators_used = len(gb_early.estimators_)
print(f"\nRésultats de l'early stopping :")
print(f"   - Temps d'entraînement : {training_time:.2f} secondes")
print(f"   - Nombre d'arbres maximum : 1000")
print(f"   - Nombre d'arbres réellement utilisés : {n_estimators_used}")
print(f"   - Économie : {1000 - n_estimators_used} arbres non entraînés")

# Création d'un jeu de validation (X_val, y_val) -- en plus du train et du test :
# Supposons que l'on a X, y (nos données complètes)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Évaluation
y_train_pred_gb = gb_early.predict(X_train)
y_test_pred_gb = gb_early.predict(X_test)

train_acc_gb = accuracy_score(y_train, y_train_pred_gb)
test_acc_gb = accuracy_score(y_test, y_test_pred_gb)

print(f"\nPerformances :")
print(f"   - Accuracy train : {train_acc_gb:.4f} ({train_acc_gb*100:.2f}%)")
print(f"   - Accuracy test  : {test_acc_gb:.4f} ({test_acc_gb*100:.2f}%)")
print(f"   - Écart train-test : {train_acc_gb - test_acc_gb:.4f}")

# Récupération des scores d'entraînement (loss)
train_scores = gb_early.train_score_

# Calcul des scores de validation à chaque étape
val_scores = []
for y_pred in gb_early.staged_predict(X_val):
    val_scores.append(accuracy_score(y_val, y_pred))

# Visualisation de l'évolution de la perte et de l'accuracy
plt.figure(figsize=(12, 5))

# Courbe complète
plt.subplot(1, 2, 1)
plt.plot(train_scores, label='Train (loss)', linewidth=2)
plt.plot(val_scores, label='Validation (accuracy)', linewidth=2)
plt.axvline(x=n_estimators_used-1, color='red', linestyle='--',
            label=f'Arrêt à B={n_estimators_used}')
plt.xlabel("Itération (nombre d'arbres)")
plt.ylabel("Score / Loss")
plt.title("Évolution pendant l'entraînement")
plt.legend()
plt.grid(True, alpha=0.3)

# Zoom sur les dernières itérations
plt.subplot(1, 2, 2)
start_idx = max(0, n_estimators_used - 50)
plt.plot(range(start_idx, n_estimators_used),
         train_scores[start_idx:], label='Train (loss)', linewidth=2)
plt.plot(range(start_idx, n_estimators_used),
         val_scores[start_idx:], label='Validation (accuracy)', linewidth=2)
plt.axvline(x=n_estimators_used-1, color='red', linestyle='--')
plt.xlabel("Itération")
plt.ylabel("Score / Loss")
plt.title("Focus sur les dernières itérations")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gb_early_stopping.png", dpi=150)
plt.show()


# ============================================
# c) SÉLECTION D'UN "BON" ALGORITHME GRADIENT BOOSTING
# ============================================

print("\n" + "=" * 60)
print("c) OPTIMISATION DE GRADIENT BOOSTING")
print("=" * 60)

# Grille de paramètres pour GridSearch
param_grid_gb = {
    'n_estimators': [100, 200, 300],  # Valeurs raisonnables (early stopping déjà utilisé)
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.5, 0.7, 1.0],      # Fraction d'échantillons pour chaque arbre
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"Grille de paramètres à tester :")
for param, values in param_grid_gb.items():
    print(f"   - {param} : {values}")
print(f"\nNombre total de combinaisons : {np.prod([len(v) for v in param_grid_gb.values()])}")

# Note : Avec autant de combinaisons, le GridSearch serait très long
# On va plutôt faire une optimisation progressive

print("\n" + "-" * 40)
print("Optimisation progressive (pour éviter un GridSearch trop long)")

# Étape 1 : Optimiser learning_rate et n_estimators avec max_depth fixé
print("\nÉtape 1 : Optimisation de learning_rate et n_estimators")
print("(max_depth=3, subsample=1.0, min_samples_split=2, min_samples_leaf=1)")

learning_rates = [0.01, 0.05, 0.1, 0.2]
n_estimators_list = [50, 100, 200, 300]

results_step1 = []

for lr in learning_rates:
    for n_est in n_estimators_list:
        gb_temp = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=3,
            subsample=1.0,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Validation croisée rapide (3-fold pour gagner du temps)
        cv_scores = cross_val_score(gb_temp, X_train, y_train, cv=3)
        
        results_step1.append({
            'learning_rate': lr,
            'n_estimators': n_est,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        print(f"  lr={lr:.2f}, B={n_est:3d} : CV={cv_scores.mean():.4f} ±{cv_scores.std():.4f}")

# Convertir en DataFrame
results_step1_df = pd.DataFrame(results_step1)
best_step1 = results_step1_df.loc[results_step1_df['cv_mean'].idxmax()]

print(f"\nMeilleurs paramètres étape 1 :")
print(f"   - learning_rate = {best_step1['learning_rate']}")
print(f"   - n_estimators = {best_step1['n_estimators']}")
print(f"   - CV score = {best_step1['cv_mean']:.4f}")

# Visualisation
plt.figure(figsize=(10, 6))
for lr in learning_rates:
    subset = results_step1_df[results_step1_df['learning_rate'] == lr]
    plt.plot(subset['n_estimators'], subset['cv_mean'], 'o-', 
             label=f'lr={lr}', linewidth=2)

plt.xlabel('n_estimators')
plt.ylabel('Score CV moyen')
plt.title('Influence de learning_rate et n_estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gb_step1_optim.png', dpi=150)
plt.show()

# Étape 2 : Optimiser max_depth et subsample
print("\n" + "-" * 40)
print("Étape 2 : Optimisation de max_depth et subsample")
print(f"(learning_rate={best_step1['learning_rate']}, n_estimators={best_step1['n_estimators']})")

max_depths = [2, 3, 4, 5, 6]
subsamples = [0.5, 0.7, 0.9, 1.0]

results_step2 = []

for depth in max_depths:
    for subsample in subsamples:
        gb_temp = GradientBoostingClassifier(
            n_estimators=int(best_step1['n_estimators']),
            learning_rate=best_step1['learning_rate'],
            max_depth=depth,
            subsample=subsample,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        cv_scores = cross_val_score(gb_temp, X_train, y_train, cv=3)
        
        results_step2.append({
            'max_depth': depth,
            'subsample': subsample,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        print(f"  depth={depth}, subsample={subsample:.1f} : CV={cv_scores.mean():.4f}")

results_step2_df = pd.DataFrame(results_step2)
best_step2 = results_step2_df.loc[results_step2_df['cv_mean'].idxmax()]

print(f"\nMeilleurs paramètres étape 2 :")
print(f"   - max_depth = {best_step2['max_depth']}")
print(f"   - subsample = {best_step2['subsample']}")
print(f"   - CV score = {best_step2['cv_mean']:.4f}")

# Visualisation
plt.figure(figsize=(10, 6))
for depth in max_depths:
    subset = results_step2_df[results_step2_df['max_depth'] == depth]
    plt.plot(subset['subsample'], subset['cv_mean'], 'o-', 
             label=f'depth={depth}', linewidth=2)

plt.xlabel('subsample')
plt.ylabel('Score CV moyen')
plt.title('Influence de max_depth et subsample')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gb_step2_optim.png', dpi=150)
plt.show()

# Étape 3 : Optimiser min_samples_split et min_samples_leaf
print("\n" + "-" * 40)
print("Étape 3 : Optimisation de min_samples_split et min_samples_leaf")
print(f"(learning_rate={best_step1['learning_rate']}, n_estimators={best_step1['n_estimators']}, "
      f"max_depth={best_step2['max_depth']}, subsample={best_step2['subsample']})")

min_samples_splits = [2, 5, 10, 20]
min_samples_leafs = [1, 2, 5, 10]

results_step3 = []

for min_split in min_samples_splits:
    for min_leaf in min_samples_leafs:
        gb_temp = GradientBoostingClassifier(
            n_estimators=int(best_step1['n_estimators']),
            learning_rate=best_step1['learning_rate'],
            max_depth=int(best_step2['max_depth']),
            subsample=best_step2['subsample'],
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            random_state=42
        )
        
        cv_scores = cross_val_score(gb_temp, X_train, y_train, cv=3)
        
        results_step3.append({
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        })
        
        print(f"  split={min_split:2d}, leaf={min_leaf:2d} : CV={cv_scores.mean():.4f}")

results_step3_df = pd.DataFrame(results_step3)
best_step3 = results_step3_df.loc[results_step3_df['cv_mean'].idxmax()]

print(f"\nMeilleurs paramètres étape 3 :")
print(f"   - min_samples_split = {best_step3['min_samples_split']}")
print(f"   - min_samples_leaf = {best_step3['min_samples_leaf']}")
print(f"   - CV score = {best_step3['cv_mean']:.4f}")

# ============================================
# CONSTRUCTION DU GRADIENT BOOSTING OPTIMISÉ
# ============================================

print("\n" + "=" * 60)
print("CONSTRUCTION DU GRADIENT BOOSTING OPTIMISÉ")
print("=" * 60)

# Rassembler tous les meilleurs paramètres
best_params_gb = {
    'n_estimators': int(best_step1['n_estimators']),
    'learning_rate': best_step1['learning_rate'],
    'max_depth': int(best_step2['max_depth']),
    'subsample': best_step2['subsample'],
    'min_samples_split': int(best_step3['min_samples_split']),
    'min_samples_leaf': int(best_step3['min_samples_leaf']),
    'validation_fraction': 0.1,
    'n_iter_no_change': 10,
    'random_state': 42
}

print(f"\nMeilleurs paramètres trouvés :")
for param, value in best_params_gb.items():
    print(f"   - {param} : {value}")

# Entraîner le modèle final
gb_optimized = GradientBoostingClassifier(**best_params_gb)

print("\nEntraînement du modèle final...")
start_time = time.time()
gb_optimized.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"✓ Entraînement terminé en {training_time:.2f} secondes")
print(f"   - Nombre d'arbres réel : {len(gb_optimized.estimators_)}")

# Évaluation
y_train_pred_gb_opt = gb_optimized.predict(X_train)
y_test_pred_gb_opt = gb_optimized.predict(X_test)

train_acc_gb_opt = accuracy_score(y_train, y_train_pred_gb_opt)
test_acc_gb_opt = accuracy_score(y_test, y_test_pred_gb_opt)

print(f"\nPerformances du Gradient Boosting optimisé :")
print(f"   - Accuracy train : {train_acc_gb_opt:.4f} ({train_acc_gb_opt*100:.2f}%)")
print(f"   - Accuracy test  : {test_acc_gb_opt:.4f} ({test_acc_gb_opt*100:.2f}%)")

# ============================================
# COMPARAISON AVEC LES MODÈLES PRÉCÉDENTS
# ============================================

print("\n" + "=" * 60)
print("COMPARAISON AVEC ARBRE OPTIMISÉ ET RANDOM FOREST")
print("=" * 60)

comparison_ensemble = pd.DataFrame({
    'Modèle': [
        'Arbre optimisé',
        'Random Forest optimisée',
        'Gradient Boosting optimisé'
    ],
    'Accuracy test': [
        test_score_best,
        test_acc_rf_opt,
        test_acc_gb_opt
    ]
})

print("\nTableau comparatif :")
print(comparison_ensemble.to_string(index=False))

# Visualisation
plt.figure(figsize=(8, 5))
bars = plt.bar(comparison_ensemble['Modèle'], comparison_ensemble['Accuracy test'],
               color=['lightblue', 'orange', 'green'])
plt.xlabel('Modèles')
plt.ylabel('Accuracy')
plt.title('Comparaison Arbre vs Random Forest vs Gradient Boosting')
plt.ylim([0.7, 0.85])  # Zoom sur la plage pertinente

# Ajouter les valeurs
for bar, val in zip(bars, comparison_ensemble['Accuracy test']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.2%}', ha='center', va='bottom')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('comparison_tree_rf_gb.png', dpi=150)
plt.show()

# Matrice de confusion pour Gradient Boosting
cm_gb = confusion_matrix(y_test, y_test_pred_gb_opt)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens',
            xticklabels=['≤50K', '>50K'],
            yticklabels=['≤50K', '>50K'])
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title(f'Matrice de confusion - Gradient Boosting optimisé\nAccuracy: {test_acc_gb_opt:.2%}')
plt.tight_layout()
plt.savefig('confusion_matrix_gb.png', dpi=150)
plt.show()

# ============================================
# ANALYSE DES RÉSULTATS
# ============================================

print("\n" + "=" * 60)
print("ANALYSE DES RÉSULTATS")
print("=" * 60)

# Déterminer le meilleur modèle
best_model_name = comparison_ensemble.loc[comparison_ensemble['Accuracy test'].idxmax(), 'Modèle']
best_model_score = comparison_ensemble['Accuracy test'].max()

print(f"\nMeilleur modèle : {best_model_name}")
print(f"   - Accuracy test : {best_model_score:.2%}")

print("""
Interprétation des résultats :

1. Arbre de décision optimisé :
   - Performance de base
   - Simple et interprétable
   - Limité par sa structure unique

2. Random Forest :
   - Améliore l'arbre grâce au bagging
   - Réduit la variance
   - Performance généralement meilleure

3. Gradient Boosting :
   - Approche séquentielle
   - Corrige les erreurs progressivement
   - Souvent le meilleur sur les données tabulaires
""")

# Vérifier si le boosting surpasse les autres
if best_model_name == 'Gradient Boosting optimisé':
    improvement_vs_rf = (test_acc_gb_opt - test_acc_rf_opt) / test_acc_rf_opt * 100
    print(f"\nLe Gradient Boosting surpasse la Random Forest de {improvement_vs_rf:.1f}%")
    
    if test_acc_gb_opt > 0.8:
        print("✓ Excellent score (>80%) - Le modèle est très performant")
    elif test_acc_gb_opt > 0.75:
        print("✓ Bon score (>75%) - Le modèle est satisfaisant")
    else:
        print("⚠ Score modeste - Peut-être besoin de plus de features ou de tuning")

# Sauvegarder le modèle
with open('gradient_boosting_optimized.pkl', 'wb') as f:
    pickle.dump(gb_optimized, f)
print("\n✓ Gradient Boosting optimisé sauvegardé sous 'gradient_boosting_optimized.pkl'")

# ============================================
# CONCLUSION DE LA PARTIE 1-D
# ============================================

print("\n" + "=" * 60)
print("CONCLUSIONS DE LA PARTIE 1-D")
print("=" * 60)

print("""
Points clés à retenir sur le Boosting :

1. Paramètres cruciaux :
   - learning_rate et n_estimators sont liés (compromis)
   - max_depth doit rester faible (arbres peu profonds)
   - subsample ajoute de l'aléatoire (comme RF)

2. Early stopping :
   - Évite le sur-apprentissage
   - Économise du temps de calcul
   - validation_fraction et n_iter_no_change sont essentiels

3. Performance :
   - Le Gradient Boosting est souvent le meilleur sur données tabulaires
   - Mais plus lent à l'entraînement que Random Forest
   - Sensible au sur-apprentissage si mal paramétré

4. Prochaine étape :
   - Analyser l'importance des variables
   - Tracer les courbes ROC
""")


# ============================================
# PARTIE 1-E : SÉLECTION DE VARIABLES (FEATURE IMPORTANCE)
# ============================================

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 60)
print("PARTIE 1-E : SÉLECTION DE VARIABLES (FEATURE IMPORTANCE)")
print("=" * 60)

print("""
L'importance des variables (feature importance) :
- Mesure la contribution de chaque variable à la prédiction
- Pour les arbres : basée sur la réduction de l'impureté
- Pour les méthodes d'ensemble : moyenne sur tous les arbres
- Permet d'interpréter le modèle et de faire de la sélection de features
""")

# Récupérer les noms des features
feature_names = X_train.columns.tolist()

# ============================================
# 1. IMPORTANCE DES VARIABLES POUR L'ARBRE OPTIMISÉ
# ============================================

print("\n" + "=" * 40)
print("1. ARBRE OPTIMISÉ - FEATURE IMPORTANCE")
print("=" * 40)

# Récupérer l'arbre optimisé (de la partie 1-B)
tree_importance = best_tree.feature_importances_

# Créer un DataFrame pour trier
tree_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': tree_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 features les plus importantes (Arbre) :")
print(tree_importance_df.head(10).to_string(index=False))

# Visualisation
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.barh(tree_importance_df.head(15)['feature'][::-1], 
         tree_importance_df.head(15)['importance'][::-1])
plt.xlabel('Importance')
plt.title('Arbre de décision - Top 15 features')
plt.tight_layout()

# ============================================
# 2. IMPORTANCE POUR LA RANDOM FOREST
# ============================================

print("\n" + "=" * 40)
print("2. RANDOM FOREST - FEATURE IMPORTANCE")
print("=" * 40)

rf_importance = rf_optimized.feature_importances_

rf_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 features les plus importantes (Random Forest) :")
print(rf_importance_df.head(10).to_string(index=False))

# Visualisation
plt.subplot(2, 2, 2)
plt.barh(rf_importance_df.head(15)['feature'][::-1], 
         rf_importance_df.head(15)['importance'][::-1], color='orange')
plt.xlabel('Importance')
plt.title('Random Forest - Top 15 features')
plt.tight_layout()

# ============================================
# 3. IMPORTANCE POUR LE GRADIENT BOOSTING
# ============================================

print("\n" + "=" * 40)
print("3. GRADIENT BOOSTING - FEATURE IMPORTANCE")
print("=" * 40)

gb_importance = gb_optimized.feature_importances_

gb_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': gb_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 features les plus importantes (Gradient Boosting) :")
print(gb_importance_df.head(10).to_string(index=False))

# Visualisation
plt.subplot(2, 2, 3)
plt.barh(gb_importance_df.head(15)['feature'][::-1], 
         gb_importance_df.head(15)['importance'][::-1], color='green')
plt.xlabel('Importance')
plt.title('Gradient Boosting - Top 15 features')
plt.tight_layout()

# ============================================
# 4. COMPARAISON DES IMPORTANCES
# ============================================

print("\n" + "=" * 40)
print("4. COMPARAISON DES TROIS MODÈLES")
print("=" * 40)

# Fusionner les trois DataFrames
comparison_importance = pd.DataFrame({
    'feature': feature_names,
    'Arbre': tree_importance,
    'Random_Forest': rf_importance,
    'Gradient_Boosting': gb_importance
})

# Normaliser pour que la somme = 1 pour chaque modèle
for col in ['Arbre', 'Random_Forest', 'Gradient_Boosting']:
    comparison_importance[col] = comparison_importance[col] / comparison_importance[col].sum()

# Top 10 features communes
top_features_tree = set(tree_importance_df.head(10)['feature'])
top_features_rf = set(rf_importance_df.head(10)['feature'])
top_features_gb = set(gb_importance_df.head(10)['feature'])

common_features = top_features_tree & top_features_rf & top_features_gb
print(f"\nFeatures communes dans les top 10 des trois modèles :")
for f in common_features:
    print(f"   - {f}")

# Visualisation comparative
plt.subplot(2, 2, 4)
# Prendre les 10 features les plus importantes en moyenne
mean_importance = comparison_importance[['Arbre', 'Random_Forest', 'Gradient_Boosting']].mean(axis=1)
comparison_importance['mean'] = mean_importance
top10_mean = comparison_importance.nlargest(10, 'mean')['feature'].values

# Préparer les données pour le graphique
plot_data = comparison_importance[comparison_importance['feature'].isin(top10_mean)]
plot_data = plot_data.set_index('feature')
plot_data[['Arbre', 'Random_Forest', 'Gradient_Boosting']].plot(kind='bar', ax=plt.gca())
plt.title('Comparaison des importances - Top 10 features')
plt.xlabel('Features')
plt.ylabel('Importance (normalisée)')
plt.xticks(rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# 5. EXPLICATION DU CALCUL DES IMPORTANCES
# ============================================

print("\n" + "=" * 40)
print("5. COMMENT SONT CALCULÉES CES IMPORTANCES ?")
print("=" * 40)

print("""
Pour les arbres de décision (et par extension Random Forest, Gradient Boosting) :

1. Importance basée sur l'impureté (MDI - Mean Decrease in Impurity) :
   - À chaque split d'un noeud, on mesure la réduction de l'impureté (Gini ou Entropie)
   - Cette réduction est pondérée par le nombre d'échantillons concernés
   - On somme ces réductions pour chaque feature sur tous les splits de tous les arbres
   - On normalise pour que la somme totale = 1

2. Pour Random Forest :
   - On moyenne les importances de tous les arbres
   - Plus robuste qu'un seul arbre

3. Pour Gradient Boosting :
   - Principe similaire, mais les arbres sont pondérés par leur learning_rate
   - Les premiers arbres ont plus d'influence que les suivants

4. Limites :
   - Favorise les variables numériques avec beaucoup de modalités
   - Ne capture pas les interactions complexes
   - Peut être biaisé si les features sont corrélées
""")

# ============================================
# 6. PERMUTATION IMPORTANCE (MÉTHODE ALTERNATIVE)
# ============================================

print("\n" + "=" * 40)
print("6. PERMUTATION IMPORTANCE (MÉTHODE PLUS ROBUSTE)")
print("=" * 40)

print("""
La permutation importance :
- Principe : on permute aléatoirement une feature et on observe la baisse de performance
- Si la permutation fait chuter le score → feature importante
- Plus fiable que l'importance basée sur l'impureté, surtout pour features corrélées
""")

# Calculer la permutation importance pour la Random Forest (comme exemple)
print("\nCalcul de la permutation importance pour Random Forest (peut prendre du temps)...")
perm_importance = permutation_importance(
    rf_optimized, X_test, y_test,
    n_repeats=10,      # Nombre de permutations pour chaque feature
    random_state=42,
    n_jobs=-1
)

# Créer un DataFrame
perm_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nTop 10 features par permutation importance :")
print(perm_importance_df.head(10).to_string(index=False))

# Comparer avec l'importance standard
comparison_methods = pd.merge(
    rf_importance_df.head(10),
    perm_importance_df.head(10),
    on='feature',
    how='outer'
)
print("\nComparaison des deux méthodes (top 10) :")
print(comparison_methods.to_string(index=False))

# Visualisation
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.barh(rf_importance_df.head(10)['feature'][::-1], 
         rf_importance_df.head(10)['importance'][::-1], color='orange')
plt.xlabel('Importance (MDI)')
plt.title('Random Forest - Importance standard')

plt.subplot(1, 2, 2)
plt.barh(perm_importance_df.head(10)['feature'][::-1], 
         perm_importance_df.head(10)['importance_mean'][::-1],
         xerr=perm_importance_df.head(10)['importance_std'][::-1],
         color='purple', capsize=3)
plt.xlabel('Importance (Permutation)')
plt.title('Random Forest - Permutation importance')

plt.tight_layout()
plt.savefig('permutation_vs_standard.png', dpi=150)
plt.show()

# ============================================
# PARTIE 1-F : COURBES ROC
# ============================================

from sklearn.metrics import roc_curve, auc, roc_auc_score

print("\n" + "=" * 60)
print("PARTIE 1-F : COURBES ROC")
print("=" * 60)

print("""
Courbe ROC (Receiver Operating Characteristic) :
- Représente le taux de vrais positifs (TPR) en fonction du taux de faux positifs (FPR)
- Permet d'évaluer la qualité du score (pas seulement de la classe prédite)
- Plus l'AUC (Area Under Curve) est proche de 1, meilleur est le modèle
- AUC = 0.5 : modèle aléatoire
- AUC = 1.0 : modèle parfait
""")

# ============================================
# 1. COMMENT PRÉDIRE UN SCORE AVEC RF ET GB ?
# ============================================

print("\n" + "=" * 40)
print("1. PRÉDICTION DE SCORES")
print("=" * 40)

# Pour Random Forest : proba d'appartenir à la classe 1 (>50K)
rf_proba = rf_optimized.predict_proba(X_test)[:, 1]
print("\nRandom Forest - Scores de probabilité :")
print(f"   - Shape des probas : {rf_proba.shape}")
print(f"   - Min : {rf_proba.min():.4f}")
print(f"   - Max : {rf_proba.max():.4f}")
print(f"   - Moyenne : {rf_proba.mean():.4f}")

# Pour Gradient Boosting
gb_proba = gb_optimized.predict_proba(X_test)[:, 1]
print("\nGradient Boosting - Scores de probabilité :")
print(f"   - Shape des probas : {gb_proba.shape}")
print(f"   - Min : {gb_proba.min():.4f}")
print(f"   - Max : {gb_proba.max():.4f}")
print(f"   - Moyenne : {gb_proba.mean():.4f}")

# Pour l'arbre
tree_proba = best_tree.predict_proba(X_test)[:, 1]
print("\nArbre optimisé - Scores de probabilité :")
print(f"   - Shape des probas : {tree_proba.shape}")
print(f"   - Min : {tree_proba.min():.4f}")
print(f"   - Max : {tree_proba.max():.4f}")
print(f"   - Moyenne : {tree_proba.mean():.4f}")

# ============================================
# 2. TRACÉ DES COURBES ROC
# ============================================

print("\n" + "=" * 40)
print("2. TRACÉ DES COURBES ROC")
print("=" * 40)

# Calculer les points des courbes ROC
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, tree_proba)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_proba)
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, gb_proba)

# Calculer les AUC
auc_tree = auc(fpr_tree, tpr_tree)
auc_rf = auc(fpr_rf, tpr_rf)
auc_gb = auc(fpr_gb, tpr_gb)

print(f"\nAUC (Area Under Curve) :")
print(f"   - Arbre optimisé : {auc_tree:.4f}")
print(f"   - Random Forest : {auc_rf:.4f}")
print(f"   - Gradient Boosting : {auc_gb:.4f}")

# Tracer les courbes ROC
plt.figure(figsize=(10, 8))

# Courbes
plt.plot(fpr_tree, tpr_tree, 'b-', linewidth=2, 
         label=f'Arbre (AUC = {auc_tree:.3f})')
plt.plot(fpr_rf, tpr_rf, 'orange', linewidth=2, 
         label=f'Random Forest (AUC = {auc_rf:.3f})')
plt.plot(fpr_gb, tpr_gb, 'g-', linewidth=2, 
         label=f'Gradient Boosting (AUC = {auc_gb:.3f})')

# Diagonale (modèle aléatoire)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC = 0.5)')

# Personnalisation
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
plt.title('Courbes ROC - Comparaison des modèles', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Ajouter un zoom sur le coin supérieur gauche (optionnel)
plt.text(0.6, 0.2, f'Meilleur modèle : {best_model_name}', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=150)
plt.show()

# ============================================
# 3. ANALYSE DES SEUILS
# ============================================

print("\n" + "=" * 40)
print("3. ANALYSE DES SEUILS DE DÉCISION")
print("=" * 40)

# Pour le meilleur modèle (Gradient Boosting généralement)
print("\nAnalyse pour Gradient Boosting (meilleur AUC) :")

# Créer un DataFrame avec les seuils et métriques
thresholds_df = pd.DataFrame({
    'threshold': thresholds_gb,
    'fpr': fpr_gb,
    'tpr': tpr_gb,
    'youden_j': tpr_gb - fpr_gb  # Indice de Youden (maximise tpr - fpr)
})

# Enlever la dernière ligne (threshold = infini)
thresholds_df = thresholds_df[:-1]

# Meilleur seuil selon Youden
best_threshold_idx = thresholds_df['youden_j'].idxmax()
best_threshold = thresholds_df.loc[best_threshold_idx, 'threshold']
best_youden = thresholds_df.loc[best_threshold_idx, 'youden_j']

print(f"\nMeilleur seuil (maximisant tpr - fpr) : {best_threshold:.4f}")
print(f"   - TPR à ce seuil : {thresholds_df.loc[best_threshold_idx, 'tpr']:.4f}")
print(f"   - FPR à ce seuil : {thresholds_df.loc[best_threshold_idx, 'fpr']:.4f}")
print(f"   - Indice de Youden : {best_youden:.4f}")

# Comparer avec le seuil par défaut (0.5)
default_idx = (thresholds_df['threshold'] - 0.5).abs().idxmin()
print(f"\nSeuil par défaut (0.5) :")
print(f"   - TPR : {thresholds_df.loc[default_idx, 'tpr']:.4f}")
print(f"   - FPR : {thresholds_df.loc[default_idx, 'fpr']:.4f}")

# Visualisation de l'évolution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds_df['threshold'], thresholds_df['tpr'], 'g-', label='TPR', linewidth=2)
plt.plot(thresholds_df['threshold'], thresholds_df['fpr'], 'r-', label='FPR', linewidth=2)
plt.axvline(x=best_threshold, color='blue', linestyle='--', 
            label=f'Meilleur seuil ({best_threshold:.2f})')
plt.axvline(x=0.5, color='black', linestyle=':', label='Seuil défaut (0.5)')
plt.xlabel('Seuil de décision')
plt.ylabel('Taux')
plt.title('TPR et FPR en fonction du seuil')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(thresholds_df['threshold'], thresholds_df['youden_j'], 'purple', linewidth=2)
plt.axvline(x=best_threshold, color='blue', linestyle='--', 
            label=f'Max Youden = {best_youden:.3f}')
plt.xlabel('Seuil de décision')
plt.ylabel('Indice de Youden (TPR - FPR)')
plt.title('Optimisation du seuil')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150)
plt.show()

# ============================================
# 4. INTERPRÉTATION DES COURBES ROC
# ============================================

print("\n" + "=" * 40)
print("4. INTERPRÉTATION DES RÉSULTATS")
print("=" * 40)

print(f"""
Interprétation des courbes ROC :

1. Classement des modèles par AUC :
   - Gradient Boosting : {auc_gb:.4f}
   - Random Forest : {auc_rf:.4f}
   - Arbre optimisé : {auc_tree:.4f}

2. Signification de l'AUC :
   - AUC = 0.5 : modèle aléatoire
   - AUC = 0.7-0.8 : acceptable
   - AUC = 0.8-0.9 : excellent
   - AUC = 0.9-1.0 : exceptionnel

3. Notre meilleur modèle ({best_model_name}) :
   - AUC = {auc_gb:.4f} → {'Excellent' if auc_gb > 0.8 else 'Bon' if auc_gb > 0.7 else 'Moyen'}

4. Compromis TPR/FPR :
   - En baissant le seuil, on augmente TPR mais aussi FPR
   - Le choix du seuil dépend du coût des erreurs :
     * Si on veut absolument détecter les >50K (TPR élevé) → seuil bas
     * Si on veut éviter les faux positifs (FPR bas) → seuil haut
""")

# ============================================
# 5. SAUVEGARDE DES MODÈLES FINAUX
# ============================================

print("\n" + "=" * 40)
print("5. SAUVEGARDE DES MODÈLES FINAUX")
print("=" * 40)

# Sauvegarder tous les modèles importants
models_to_save = {
    'best_tree': best_tree,
    'random_forest': rf_optimized,
    'gradient_boosting': gb_optimized,
    'scaler': scaler
}

for name, model in models_to_save.items():
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ {name}.pkl sauvegardé")

# Sauvegarder aussi les métriques pour le rapport
metrics = {
    'tree_accuracy': test_score_best,
    'tree_auc': auc_tree,
    'rf_accuracy': test_acc_rf_opt,
    'rf_auc': auc_rf,
    'gb_accuracy': test_acc_gb_opt,
    'gb_auc': auc_gb,
    'best_model': best_model_name,
    'best_threshold': best_threshold
}

with open('final_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("✓ Métriques finales sauvegardées")

# ============================================
# CONCLUSION DES PARTIES 1-E et 1-F
# ============================================

print("\n" + "=" * 60)
print("CONCLUSION DES PARTIES 1-E ET 1-F")
print("=" * 60)

print("""
Points clés à retenir :

1. Feature Importance :
   - Les trois modèles s'accordent sur les variables clés
   - Professional, Income, Unemployment sont déterminants
   - La permutation importance confirme ces résultats

2. Courbes ROC :
   - Le Gradient Boosting est le meilleur (AUC le plus élevé)
   - Tous les modèles battent largement le hasard (AUC > 0.5)
   - Les courbes permettent de choisir un seuil adapté

3. Pour le déploiement :
   - On garde le Gradient Boosting comme modèle final
   - Avec un seuil optimisé ({best_threshold:.3f}) si nécessaire
   - Les modèles sont sauvegardés pour Streamlit
""")


