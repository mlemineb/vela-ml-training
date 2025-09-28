# -*- coding: utf-8 -*-
# %% [markdown]
"""
# 🎓 Projet Capstone - Classification BAC Mauritanie 2022
## 🤝 Partie 2 : Préparation des Données et Modélisation IA

### 🔄 Continuité du Projet
Vous avez terminé l'exploration des données dans la Partie 1.
Maintenant, nous allons préparer ces données pour l'Intelligence Artificielle !

**Rappel des Découvertes Partie 1 :**
- Répartition Admis/Sessionnaire
- Différences par genre
- Disparités régionales (wilayas)

---
*Projet Capstone - SupNum Nouakchott - Formation IA & Machine Learning*
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("🛠️ PARTIE 2 : PRÉPARATION ET MODÉLISATION IA")
print("=" * 50)

# %% [markdown]
"""
## 🛠️ SECTION 3: Préparation des Données pour l'IA (15 min)

### 🔄 Rappel Session 3
Lors de la Session 3, vous avez appris que les algorithmes ML ont besoin de données **propres** et **numériques**.

### 🎯 Mission
Transformer nos données mauritaniennes pour que l'IA puisse les comprendre !

### ❓ Questions Techniques Binôme
Discutez ces questions avant de voir le code :

**Question 1 :** Pourquoi exclure les 'Absents' de notre analyse ?
- Réponse Binôme : ___________________________________

**Question 2 :** Comment l'IA peut-elle comprendre 'Masculin/Féminin' ?
- Réponse Binôme : ___________________________________

**Question 3 :** Faut-il garder tous les âges ou filtrer les cas extrêmes ?
- Réponse Binôme : ___________________________________
"""

# %%
# Chargement et préparation des données
print("📊 CHARGEMENT ET PRÉPARATION DES DONNÉES")
print("-" * 45)

# IMPORTANT: Remplacez le chemin par le vôtre
data_path = "dataset/BAC - 2022.xlsx"

try:
    # Chargement du dataset
    df = pd.read_excel(data_path)
    print(f"✅ Dataset chargé : {len(df)} élèves")
    
    # Filtrage : garder seulement ADMIS, SESSIONNAIRE et AJOURNÉ
    df_work = df[df['Decision'].isin(['Admis', 'Sessionnaire', 'Ajourné'])].copy()
    print(f"📊 Après filtrage : {len(df_work)} élèves (Admis + Sessionnaire + Ajourné)")
    
    # Création de la variable cible CORRIGÉE
    df_work['Succes'] = (df_work['Decision'].isin(['Admis', 'Sessionnaire'])).astype(int)
    
    # Vérification des colonnes importantes
    colonnes_importantes = ['Sexe', 'Age', 'Serie', 'Wilaya', 'Etablissement', 'Succes']
    colonnes_disponibles = [col for col in colonnes_importantes if col in df_work.columns]
    
    print(f"✅ Colonnes disponibles : {colonnes_disponibles}")
    print(f"✅ Variable cible : Réussite = 1 (Admis + Sessionnaire), Échec = 0 (Ajourné)")
    
except FileNotFoundError:
    print("❌ Fichier non trouvé ! Vérifiez le chemin du dataset.")

# %% [markdown]
"""
### 🧹 Étape 1: Nettoyage des Données

**Tâches à Répartir dans le Binôme :**
- **Personne A :** Vérifier les valeurs manquantes
- **Personne B :** Analyser les valeurs aberrantes
- **Ensemble :** Décider des actions à prendre
"""

# %%
# Analyse de la qualité des données
print("🔍 ANALYSE DE LA QUALITÉ DES DONNÉES")
print("-" * 40)

if 'df_work' in locals():
    # Vérification des valeurs manquantes
    print("📋 Valeurs manquantes par colonne :")
    missing_data = df_work[colonnes_disponibles].isnull().sum()
    for col, missing in missing_data.items():
        pct_missing = (missing / len(df_work)) * 100
        print(f"  {col}: {missing} ({pct_missing:.1f}%)")
    
    # Analyse des âges
    if 'Age' in df_work.columns:
        print(f"\n📊 Analyse des âges :")
        print(f"  Âge minimum : {df_work['Age'].min()}")
        print(f"  Âge maximum : {df_work['Age'].max()}")
        print(f"  Âge moyen : {df_work['Age'].mean():.1f}")
        
        # Détection des âges aberrants
        ages_extremes = df_work[(df_work['Age'] < 16) | (df_work['Age'] > 25)]
        print(f"  Âges extrêmes (<16 ou >25) : {len(ages_extremes)} élèves")
    
    # Répartition des séries
    if 'Serie' in df_work.columns:
        print(f"\n📚 Répartition par série :")
        serie_counts = df_work['Serie'].value_counts()
        for serie, count in serie_counts.items():
            pct = (count / len(df_work)) * 100
            print(f"  {serie}: {count} ({pct:.1f}%)")

# %% [markdown]
"""
### 🔧 Étape 2: Encodage des Variables Catégorielles

**Rappel Session 3 :** Les algorithmes ML ne comprennent que les nombres !

**Question Binôme :** Quelle méthode d'encodage choisir ?
- **Label Encoding :** Masculin=0, Féminin=1
- **One-Hot Encoding :** Créer des colonnes binaires
- Réponse Binôme : ___________________________________
"""

# %%
# Encodage des variables catégorielles
print("🔄 ENCODAGE DES VARIABLES CATÉGORIELLES")
print("-" * 42)

if 'df_work' in locals():
    # Création d'une copie pour l'encodage
    df_encoded = df_work.copy()
    
    # Dictionnaire pour stocker les encodeurs
    encoders = {}
    
    # Encodage des variables catégorielles
    categorical_vars = ['Sexe', 'Serie', 'Wilaya', 'Etablissement']
    
    for var in categorical_vars:
        if var in df_encoded.columns:
            # Création de l'encodeur
            le = LabelEncoder()
            
            # Gestion des valeurs manquantes
            mask = df_encoded[var].notna()
            
            if mask.sum() > 0:  # S'il y a des valeurs non-nulles
                # Encodage
                df_encoded.loc[mask, f'{var}_encoded'] = le.fit_transform(df_encoded.loc[mask, var])
                
                # Stockage de l'encodeur
                encoders[var] = le
                
                # Affichage du mapping
                print(f"\n🔤 Encodage {var} :")
                for i, label in enumerate(le.classes_):
                    print(f"  {label} → {i}")
            else:
                print(f"❌ {var}: Aucune valeur à encoder")
    
    # Sélection des variables finales pour le modèle
    feature_columns = ['Age'] + [f'{var}_encoded' for var in categorical_vars 
                                if f'{var}_encoded' in df_encoded.columns]
    
    print(f"\n✅ Variables finales pour l'IA : {feature_columns}")

# %% [markdown]
"""
### 📊 Étape 3: Préparation du Dataset Final

**Question Binôme :** Comment diviser nos données ?
- **Entraînement :** Pour apprendre à l'IA
- **Test :** Pour évaluer les performances
- Quelle proportion recommandez-vous ? _______________
"""

# %%
# Préparation du dataset final
print("📊 PRÉPARATION DU DATASET FINAL")
print("-" * 35)

if 'df_encoded' in locals() and feature_columns:
    # Suppression des lignes avec des valeurs manquantes
    df_final = df_encoded.dropna(subset=feature_columns + ['Succes'])
    
    print(f"📊 Dataset final : {len(df_final)} élèves")
    
    # Séparation des features (X) et de la cible (y)
    X = df_final[feature_columns]
    y = df_final['Succes']
    
    print(f"✅ Variables prédictives (X) : {X.shape}")
    print(f"✅ Variable cible (y) : {y.shape}")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 DIVISION DES DONNÉES :")
    print(f"  Entraînement : {len(X_train)} élèves ({len(X_train)/len(df_final)*100:.1f}%)")
    print(f"  Test : {len(X_test)} élèves ({len(X_test)/len(df_final)*100:.1f}%)")
    
    # Vérification de l'équilibre
    print(f"\n⚖️ ÉQUILIBRE DES CLASSES :")
    print(f"  Entraînement - Admis : {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"  Test - Admis : {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

# %% [markdown]
"""
## 🤖 SECTION 4: IA en Action - Tester 4 Algorithmes (25 min)

### 🧠 Rappel Session 3
Vous avez appris 4 algorithmes de classification :
1. **Logistic Regression** - Simple et interprétable
2. **Decision Tree** - Règles claires
3. **Random Forest** - Performance élevée
4. **Naive Bayes** - Probabilités

### 🎯 Mission
Entraîner ces 4 algorithmes sur nos données mauritaniennes et voir lequel performe le mieux !

### ❓ Questions de Modélisation Binôme
Avant de voir les résultats, faites vos prédictions :

**Question 1 :** Quel algorithme donnera la meilleure précision ?
- Votre Prédiction : ___________________________________

**Question 2 :** Quelle variable sera la plus importante ?
- Votre Prédiction : ___________________________________

**Question 3 :** Attendez-vous des surprises dans les résultats ?
- Votre Prédiction : ___________________________________
"""

# %%
# Entraînement des 4 algorithmes
print("🤖 ENTRAÎNEMENT DES 4 ALGORITHMES IA")
print("-" * 40)

if 'X_train' in locals():
    # Dictionnaire pour stocker les modèles et résultats
    models = {}
    results = {}
    
    # 1. Logistic Regression
    print("🔄 Entraînement Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    models['Logistic Regression'] = lr
    results['Logistic Regression'] = {
        'accuracy': lr_accuracy,
        'predictions': lr_pred
    }
    
    # 2. Decision Tree
    print("🌳 Entraînement Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    models['Decision Tree'] = dt
    results['Decision Tree'] = {
        'accuracy': dt_accuracy,
        'predictions': dt_pred
    }
    
    # 3. Random Forest
    print("🌲 Entraînement Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'accuracy': rf_accuracy,
        'predictions': rf_pred
    }
    
    # 4. Naive Bayes
    print("🧮 Entraînement Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    
    models['Naive Bayes'] = nb
    results['Naive Bayes'] = {
        'accuracy': nb_accuracy,
        'predictions': nb_pred
    }
    
    print("✅ Tous les modèles entraînés !")

# %% [markdown]
"""
### 📊 Comparaison des Performances

**Question Binôme :** Regardez les résultats ci-dessous et répondez :
- Quel algorithme performe le mieux ? _______________
- Êtes-vous surpris par les résultats ? _______________
- Lequel choisiriez-vous pour votre école ? _______________
"""

# %%
# Comparaison des performances
print("📊 COMPARAISON DES PERFORMANCES")
print("-" * 35)

if 'results' in locals():
    # Création du tableau de comparaison
    comparison_data = {
        'Algorithme': list(results.keys()),
        'Précision (%)': [results[model]['accuracy'] * 100 for model in results.keys()],
        'Interprétabilité': ['Élevée', 'Très élevée', 'Moyenne', 'Moyenne'],
        'Vitesse': ['Rapide', 'Rapide', 'Moyenne', 'Très rapide']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Précision (%)', ascending=False)
    
    print("🏆 CLASSEMENT DES ALGORITHMES :")
    print(df_comparison.to_string(index=False))
    
    # Graphique de comparaison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique en barres des précisions
    colors = ['gold', 'silver', '#CD7F32', 'lightblue']  # Or, Argent, Bronze, Bleu
    bars = ax1.bar(df_comparison['Algorithme'], df_comparison['Précision (%)'], 
                   color=colors[:len(df_comparison)], alpha=0.8)
    ax1.set_title('🏆 Précision des Algorithmes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Précision (%)')
    ax1.set_ylim(0, 100)
    
    # Ajout des valeurs sur les barres
    for bar, value in zip(bars, df_comparison['Précision (%)']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Rotation des labels
    ax1.tick_params(axis='x', rotation=45)
    
    # Matrice de confusion pour le meilleur modèle
    best_model_name = df_comparison.iloc[0]['Algorithme']
    best_predictions = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Sessionnaire', 'Admis'],
                yticklabels=['Sessionnaire', 'Admis'])
    ax2.set_title(f'🎯 Matrice de Confusion\n{best_model_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Prédiction')
    ax2.set_ylabel('Réalité')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🥇 GAGNANT : {best_model_name} avec {df_comparison.iloc[0]['Précision (%)']:.1f}% de précision")

# %% [markdown]
"""
### 🌳 Analyse de l'Arbre de Décision

**Pourquoi regarder l'arbre ?** Il nous donne des **règles claires** que les enseignants peuvent comprendre !

**Question Binôme :** Quelles règles découvrez-vous dans l'arbre ?
- Règle 1 : ___________________________________
- Règle 2 : ___________________________________
- Règle 3 : ___________________________________
"""

# %%
# Visualisation de l'arbre de décision
print("🌳 ANALYSE DE L'ARBRE DE DÉCISION")
print("-" * 35)

if 'Decision Tree' in models:
    # Visualisation de l'arbre (simplifié)
    plt.figure(figsize=(20, 12))
    plot_tree(models['Decision Tree'], 
              feature_names=feature_columns,
              class_names=['Sessionnaire', 'Admis'],
              filled=True, 
              rounded=True,
              fontsize=10,
              max_depth=3)  # Limiter la profondeur pour la lisibilité
    
    plt.title('🌳 Arbre de Décision - Règles de Prédiction BAC', 
              fontsize=16, fontweight='bold', pad=20)
    plt.show()
    
    # Importance des variables
    if hasattr(models['Decision Tree'], 'feature_importances_'):
        importance_data = {
            'Variable': feature_columns,
            'Importance': models['Decision Tree'].feature_importances_
        }
        df_importance = pd.DataFrame(importance_data)
        df_importance = df_importance.sort_values('Importance', ascending=False)
        
        print("📊 IMPORTANCE DES VARIABLES :")
        for _, row in df_importance.iterrows():
            print(f"  {row['Variable']}: {row['Importance']:.3f}")
        
        # Graphique d'importance
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_importance['Variable'], df_importance['Importance'], 
                      color='skyblue', alpha=0.8)
        plt.title('📊 Importance des Variables dans la Prédiction', 
                  fontsize=14, fontweight='bold')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        
        # Ajout des valeurs sur les barres
        for bar, value in zip(bars, df_importance['Importance']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
"""
### 💡 Espace Résultats Binôme

Après avoir analysé les 4 algorithmes, notez vos découvertes :

**Meilleur Algorithme :** 
- Réponse : ___________________________________

**Précision Obtenue :** 
- Réponse : ___________________________________

**Variable la Plus Importante :** 
- Réponse : ___________________________________

**Règle Découverte dans l'Arbre :** 
- Réponse : ___________________________________

**Surprise dans les Résultats :** 
- Réponse : ___________________________________

### 🎯 Questions d'Interprétation

**Question 1 :** Ces résultats confirment-ils vos hypothèses de la Partie 1 ?
- Réponse Binôme : ___________________________________

**Question 2 :** Quelle règle de l'arbre pourriez-vous appliquer dans votre classe ?
- Réponse Binôme : ___________________________________

**Question 3 :** Comment expliquer ces résultats à un directeur d'école ?
- Réponse Binôme : ___________________________________
"""

# %%
print("✅ SECTION 4 TERMINÉE - MODÉLISATION IA")
print("=" * 45)
print("🎯 Vous avez maintenant :")
print("  ✓ Préparé les données pour l'IA")
print("  ✓ Entraîné 4 algorithmes différents")
print("  ✓ Comparé leurs performances")
print("  ✓ Découvert des règles de prédiction")
print("\n🚀 Prochaine étape : Interpréter ces résultats pour l'éducation mauritanienne !")
print("=" * 45)
