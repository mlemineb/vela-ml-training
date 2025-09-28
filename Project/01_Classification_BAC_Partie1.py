# -*- coding: utf-8 -*-
# %% [markdown]
"""
# 🎓 Projet Capstone - Classification BAC Mauritanie 2022
## 🤝 Travail en Binômes - Assistant IA pour l'Éducation

### 🎯 Contexte du Projet
**Objectif :** Créer un Assistant IA qui aide les enseignants mauritaniens à prédire la réussite de leurs élèves

**Compétences Acquises à Utiliser :**
- ✅ **Session 1 :** Concepts IA/ML + Prédiction BAC
- ✅ **Session 2 :** Régression (prix maisons) 
- ✅ **Session 3 :** Classification (Titanic, algorithmes)
- ✅ **Session 4 :** Clustering (segmentation)
- ✅ **Session 5 :** Deep Learning (vision, YOLO)

**Format :** Binômes - 2 personnes par équipe
**Durée :** 45 minutes pour cette composante Classification
**Composante :** 1/4 du projet final

---
*Projet Capstone - SupNum Nouakchott - Formation IA & Machine Learning*
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("🎓 PROJET CAPSTONE - CLASSIFICATION BAC MAURITANIE 2022")
print("=" * 60)
print("🤝 Travail en Binômes - Assistant IA pour l'Éducation")
print("=" * 60)

# %% [markdown]
"""
## 📋 SECTION 0: Organisation du Binôme (5 min)

### ❓ Questions d'Organisation
Avant de commencer, organisez-vous en binôme :

**Question 1 :** Qui fait quoi dans votre binôme ?
- Personne A : _________________________________
- Personne B : _________________________________

**Question 2 :** Comment allez-vous vous répartir le travail ?
- Réponse : ___________________________________

**Question 3 :** Quel est votre objectif commun pour ce projet ?
- Réponse : ___________________________________

### 🎯 Rappel de la Mission
Vous allez créer un système qui prédit si un élève va **RÉUSSIR son BAC** 
(Admis OU Sessionnaire) ou **ÉCHOUER** (Ajourné), en utilisant seulement 
des informations disponibles AVANT les examens.
"""

# %%
# Chargement des données
print("📊 CHARGEMENT DES DONNÉES BAC MAURITANIE 2022")
print("-" * 50)

# IMPORTANT: Remplacez le chemin par le vôtre
data_path = "dataset/BAC - 2022.xlsx"

try:
    # Chargement du dataset
    df = pd.read_excel(data_path)
    print(f"✅ Dataset chargé avec succès !")
    print(f"📊 Nombre de lignes : {len(df)}")
    print(f"📊 Nombre de colonnes : {len(df.columns)}")
    
    # Affichage des premières lignes
    print("\n🔍 Aperçu des données :")
    print(df.head())
    
except FileNotFoundError:
    print("❌ Fichier non trouvé ! Vérifiez le chemin du dataset.")
    print("💡 Conseil : Placez le fichier 'BAC - 2022.xlsx' dans le dossier 'dataset/'")

# %% [markdown]
"""
## 📚 SECTION 1: Notre Mission - Prédire la Réussite (10 min)

### 🎯 Objectif Pédagogique
Utiliser la **Classification** (Session 3) sur des données réelles mauritaniennes 
pour aider les enseignants à identifier les élèves à risque d'échec.

### 🇲🇷 Impact pour la Mauritanie
Imaginez pouvoir dire à un enseignant dès septembre : 
*"Cet élève a 80% de chances d'échouer au BAC (Ajourné)"*

### ❓ Questions de Réflexion Binôme
Discutez ensemble et notez vos réponses :

**Question 1 :** En tant qu'enseignants, quels signes vous alertent qu'un élève risque d'échouer ?
- Réponse Binôme : ___________________________________

**Question 2 :** Pensez-vous pouvoir prédire la réussite AVANT les premières notes ?
- Réponse Binôme : ___________________________________

**Question 3 :** Quel serait l'impact si vous pouviez identifier les élèves à risque dès septembre ?
- Réponse Binôme : ___________________________________

### 💡 Hypothèses à Tester
Avant de voir les données, formulez vos hypothèses :

**Hypothèse 1 :** Les filles réussissent-elles mieux que les garçons ?
- Votre Prédiction : ___________________________________

**Hypothèse 2 :** Certaines régions (wilayas) ont-elles de meilleurs résultats ?
- Votre Prédiction : ___________________________________

**Hypothèse 3 :** L'âge influence-t-il la réussite au BAC ?
- Votre Prédiction : ___________________________________
"""

# %%
# Exploration initiale des données
print("🔍 EXPLORATION INITIALE DES DONNÉES")
print("-" * 40)

# Informations générales sur le dataset
print("📊 Informations générales :")
print(f"Nombre total d'élèves : {len(df)}")
print(f"Colonnes disponibles : {list(df.columns)}")

# Vérification des valeurs de la colonne Decision
print("\n🎯 Analyse de la colonne 'Decision' :")
if 'Decision' in df.columns:
    decision_counts = df['Decision'].value_counts()
    print(decision_counts)
    
    # Calcul des pourcentages
    decision_pct = df['Decision'].value_counts(normalize=True) * 100
    print("\n📊 Répartition en pourcentages :")
    for decision, pct in decision_pct.items():
        print(f"{decision}: {pct:.1f}%")
else:
    print("❌ Colonne 'Decision' non trouvée dans le dataset")

# Affichage des colonnes importantes
print("\n📋 Colonnes importantes pour notre analyse :")
important_cols = ['Sexe', 'Age', 'Serie', 'Wilaya', 'Etablissement', 'Decision']
for col in important_cols:
    if col in df.columns:
        print(f"✅ {col}: {df[col].nunique()} valeurs uniques")
    else:
        print(f"❌ {col}: Non trouvée")

# %% [markdown]
"""
## 📊 SECTION 2: Exploration - Découvrir les Patterns (20 min)

### 🔍 Rappel Session 1
Lors de la Session 1, vous avez appris l'importance de l'**analyse exploratoire** 
des données. C'est exactement ce que nous allons faire maintenant !

### 🎯 Mission
Comprendre qui **RÉUSSIT** (Admis + Sessionnaire) versus qui **ÉCHOUE** (Ajourné).

### ❓ Questions d'Analyse Binôme
Analysez les graphiques ci-dessous et répondez ensemble :
"""

# %%
# Préparation des données pour l'analyse
print("🛠️ PRÉPARATION DES DONNÉES POUR L'ANALYSE")
print("-" * 45)

# Filtrage : garder seulement ADMIS, SESSIONNAIRE et AJOURNÉ
if 'Decision' in df.columns:
    # Affichage des valeurs uniques dans Decision
    print("📋 Valeurs dans la colonne Decision :")
    print(df['Decision'].unique())
    
    # Filtrage pour garder seulement les résultats définitifs
    df_filtered = df[df['Decision'].isin(['Admis', 'Sessionnaire', 'Ajourné'])].copy()
    
    print(f"\n📊 Données après filtrage :")
    print(f"Nombre d'élèves ADMIS : {len(df_filtered[df_filtered['Decision'] == 'Admis'])}")
    print(f"Nombre d'élèves SESSIONNAIRE : {len(df_filtered[df_filtered['Decision'] == 'Sessionnaire'])}")
    print(f"Nombre d'élèves AJOURNÉ : {len(df_filtered[df_filtered['Decision'] == 'Ajourné'])}")
    print(f"Total analysé : {len(df_filtered)} élèves")
    
    # Création de la variable cible binaire CORRIGÉE
    df_filtered['Succes'] = (df_filtered['Decision'].isin(['Admis', 'Sessionnaire'])).astype(int)
    print(f"\n✅ Variable cible créée : Succès = 1 (Admis + Sessionnaire), Échec = 0 (Ajourné)")
    
    # Vérification
    reussite_count = df_filtered['Succes'].sum()
    echec_count = len(df_filtered) - reussite_count
    print(f"📊 Vérification : {reussite_count} réussites, {echec_count} échecs")
    
else:
    print("❌ Impossible de filtrer : colonne 'Decision' manquante")

# %% [markdown]
"""
### 📊 Graphique 1: Répartition Générale RÉUSSITE vs ÉCHEC

**Question Binôme 1 :** Combien d'élèves RÉUSSISSENT leur BAC ? Quel pourcentage ?
- Votre Réponse : ___________________________________

**Question Binôme 2 :** Ce taux de réussite globale vous surprend-il ?
- Votre Réponse : ___________________________________
"""

# %%
# Graphique 1: Répartition générale
if 'Decision' in df.columns and len(df_filtered) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique en barres - Détaillé
    decision_counts = df_filtered['Decision'].value_counts()
    colors = ['green', 'orange', 'red']
    bars = ax1.bar(decision_counts.index, decision_counts.values, color=colors, alpha=0.7)
    ax1.set_title('📊 Répartition Détaillée des Résultats BAC', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Nombre d\'élèves')
    
    # Ajout des valeurs sur les barres
    for bar, value in zip(bars, decision_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Graphique en camembert - Réussite vs Échec
    succes_counts = df_filtered['Succes'].value_counts()
    labels = ['ÉCHEC (Ajourné)', 'RÉUSSITE (Admis + Sessionnaire)']
    colors_binary = ['red', 'green']
    ax2.pie(succes_counts.values, labels=labels, autopct='%1.1f%%', 
            colors=colors_binary, startangle=90)
    ax2.set_title('📊 RÉUSSITE vs ÉCHEC au BAC', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    total = len(df_filtered)
    reussite = df_filtered['Succes'].sum()
    echec = total - reussite
    
    print(f"📊 STATISTIQUES GÉNÉRALES :")
    print(f"Total élèves analysés : {total}")
    print(f"RÉUSSITE (Admis + Sessionnaire) : {reussite} ({reussite/total*100:.1f}%)")
    print(f"ÉCHEC (Ajourné) : {echec} ({echec/total*100:.1f}%)")

# %% [markdown]
"""
### 📊 Graphique 2: Performance par Genre

**Question Binôme 3 :** Les filles ou les garçons ont-ils plus de chances de réussir le BAC ?
- Votre Réponse : ___________________________________

**Question Binôme 4 :** Cette différence vous surprend-elle ? Pourquoi ?
- Votre Réponse : ___________________________________
"""

# %%
# Graphique 2: Performance par Genre
if 'Sexe' in df_filtered.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Tableau croisé détaillé
    gender_decision = pd.crosstab(df_filtered['Sexe'], df_filtered['Decision'])
    
    # Graphique en barres groupées - Détaillé
    gender_decision.plot(kind='bar', ax=ax1, color=['red', 'green', 'orange'], alpha=0.7)
    ax1.set_title('📊 Résultats Détaillés par Genre', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Genre')
    ax1.set_ylabel('Nombre d\'élèves')
    ax1.legend(title='Résultat')
    ax1.tick_params(axis='x', rotation=0)
    
    # Graphique en pourcentages - Réussite/Échec
    gender_success = pd.crosstab(df_filtered['Sexe'], df_filtered['Succes'])
    gender_success.columns = ['ÉCHEC', 'RÉUSSITE']
    gender_success_pct = pd.crosstab(df_filtered['Sexe'], df_filtered['Succes'], normalize='index') * 100
    gender_success_pct.columns = ['ÉCHEC', 'RÉUSSITE']
    
    gender_success_pct.plot(kind='bar', ax=ax2, color=['red', 'green'], alpha=0.7)
    ax2.set_title('📊 Taux de Réussite par Genre', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Genre')
    ax2.set_ylabel('Pourcentage')
    ax2.legend(title='Résultat')
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques détaillées
    print("📊 ANALYSE PAR GENRE :")
    print("Résultats détaillés :")
    print(gender_decision)
    print("\nTaux de réussite :")
    print(gender_success_pct.round(1))
    
    # Calcul du taux de réussite par genre
    for genre in df_filtered['Sexe'].unique():
        subset = df_filtered[df_filtered['Sexe'] == genre]
        taux_reussite = subset['Succes'].mean() * 100
        print(f"Taux RÉUSSITE {genre} : {taux_reussite:.1f}%")

# %% [markdown]
"""
### 📊 Graphique 3: Performance par Wilaya (Région)

**Question Binôme 5 :** Votre wilaya d'origine performe-t-elle bien ?
- Votre Réponse : ___________________________________

**Question Binôme 6 :** Quelles sont les 3 meilleures wilayas ? Les 3 moins bonnes ?
- Meilleures : ___________________________________
- Moins bonnes : ___________________________________

**Question Binôme 7 :** Comment expliquez-vous ces différences régionales ?
- Votre Réponse : ___________________________________
"""

# %%
# Graphique 3: Performance par Wilaya
if 'Wilaya' in df_filtered.columns:
    # Calcul du taux de réussite par wilaya
    wilaya_stats = df_filtered.groupby('Wilaya').agg({
        'Succes': ['count', 'sum', 'mean']
    }).round(3)
    
    wilaya_stats.columns = ['Total_Eleves', 'Nb_Reussite', 'Taux_Reussite']
    wilaya_stats = wilaya_stats.sort_values('Taux_Reussite', ascending=False)
    
    # Filtrer les wilayas avec au moins 10 élèves pour éviter les biais
    wilaya_stats_filtered = wilaya_stats[wilaya_stats['Total_Eleves'] >= 10]
    
    # Graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top 10 des wilayas
    top_wilayas = wilaya_stats_filtered.head(10)
    bars1 = ax1.bar(range(len(top_wilayas)), top_wilayas['Taux_Reussite'] * 100, 
                    color='green', alpha=0.7)
    ax1.set_title('🏆 Top 10 des Wilayas - Taux de Réussite BAC', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Wilaya')
    ax1.set_ylabel('Taux RÉUSSITE (%)')
    ax1.set_xticks(range(len(top_wilayas)))
    ax1.set_xticklabels(top_wilayas.index, rotation=45, ha='right')
    
    # Ajout des valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars1, top_wilayas['Taux_Reussite'] * 100)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Bottom 10 des wilayas
    bottom_wilayas = wilaya_stats_filtered.tail(10)
    bars2 = ax2.bar(range(len(bottom_wilayas)), bottom_wilayas['Taux_Reussite'] * 100, 
                    color='red', alpha=0.7)
    ax2.set_title('📉 Bottom 10 des Wilayas - Taux de Réussite BAC', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Wilaya')
    ax2.set_ylabel('Taux RÉUSSITE (%)')
    ax2.set_xticks(range(len(bottom_wilayas)))
    ax2.set_xticklabels(bottom_wilayas.index, rotation=45, ha='right')
    
    # Ajout des valeurs sur les barres
    for i, (bar, value) in enumerate(zip(bars2, bottom_wilayas['Taux_Reussite'] * 100)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    print("📊 STATISTIQUES PAR WILAYA (min 10 élèves) :")
    print(f"Meilleure wilaya : {wilaya_stats_filtered.index[0]} ({wilaya_stats_filtered.iloc[0]['Taux_Reussite']*100:.1f}%)")
    print(f"Moins bonne wilaya : {wilaya_stats_filtered.index[-1]} ({wilaya_stats_filtered.iloc[-1]['Taux_Reussite']*100:.1f}%)")
    print(f"Écart : {(wilaya_stats_filtered.iloc[0]['Taux_Reussite'] - wilaya_stats_filtered.iloc[-1]['Taux_Reussite'])*100:.1f} points")

# %% [markdown]
"""
### 💡 Espace Découvertes Binôme

Après avoir analysé ces graphiques, notez vos découvertes :

**Découverte Surprenante 1 :** 
- Réponse : ___________________________________

**Découverte Surprenante 2 :** 
- Réponse : ___________________________________

**Hypothèse à Tester avec l'IA :** 
- Réponse : ___________________________________

**Variable qui Semble la Plus Importante :**
- Réponse : ___________________________________

### 🎯 Préparation pour la Suite
Vous avez maintenant une bonne compréhension des données ! 
Dans la prochaine section, nous allons préparer ces données 
pour les algorithmes d'Intelligence Artificielle.

**Question Finale :** Êtes-vous prêts à entraîner votre première IA sur des données mauritaniennes ?
- Réponse Binôme : ___________________________________
"""

# %%
print("✅ SECTION 2 TERMINÉE - EXPLORATION DES DONNÉES")
print("=" * 50)
print("🎯 Prochaine étape : Préparation des données pour l'IA")
print("🤖 Vous allez bientôt entraîner 4 algorithmes différents !")
print("=" * 50)
