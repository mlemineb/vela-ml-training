# -*- coding: utf-8 -*-
# %% [markdown]
"""
# 📚 Guide des Variables - Dataset BAC Mauritanie 2022
## 🎯 Comprendre les Données Avant la Classification

### 🚨 IMPORTANT : Pourquoi NE PAS Utiliser les Notes ?
**Règle d'Or en Machine Learning :** On ne peut pas utiliser des informations qu'on n'aura pas au moment de la prédiction !

**Exemple concret :**
- 🎯 **Objectif :** Prédire en septembre qui risque d'échouer
- ❌ **Problème :** Les notes ne sont connues qu'après les examens
- ✅ **Solution :** Utiliser seulement les caractéristiques disponibles AVANT les examens

---
*Guide Variables - Projet Capstone - SupNum Nouakchott*
"""

# %%
import pandas as pd
import numpy as np

print("📚 GUIDE DES VARIABLES - BAC MAURITANIE 2022")
print("=" * 50)

# %% [markdown]
"""
## 📋 DICTIONNAIRE DES VARIABLES

### 🔍 Variables Démographiques (Utilisables pour Prédiction)
Ces informations sont connues AVANT les examens :
"""

# %%
# Dictionnaire des variables utilisables
variables_utilisables = {
    'Variable': ['NumDossier', 'Sexe', 'Age', 'Prenom', 'Serie', 'DateNaissance', 
                 'LieuNaissance', 'Centre', 'Etablissement', 'Wilaya'],
    'Description': [
        'Numéro unique du dossier candidat',
        'Genre du candidat (Masculin/Féminin)',
        'Âge du candidat au moment du BAC',
        'Prénom du candidat',
        'Filière choisie (SN, M, LO, TM, LM)',
        'Date de naissance du candidat',
        'Lieu de naissance du candidat',
        'Centre d\'examen assigné',
        'École d\'origine du candidat',
        'Région (wilaya) d\'origine'
    ],
    'Type': ['Numérique', 'Catégorielle', 'Numérique', 'Texte', 'Catégorielle',
             'Date', 'Texte', 'Catégorielle', 'Catégorielle', 'Catégorielle'],
    'Utilité_Prédiction': ['Identifiant', 'Très utile', 'Très utile', 'Peu utile', 'Très utile',
                          'Peu utile', 'Utile', 'Utile', 'Très utile', 'Très utile']
}

df_variables = pd.DataFrame(variables_utilisables)
print("✅ VARIABLES UTILISABLES POUR LA PRÉDICTION :")
print(df_variables.to_string(index=False))

# %% [markdown]
"""
### 🚨 Variables à NE PAS Utiliser (Notes)
Ces informations ne sont connues qu'APRÈS les examens :
"""

# %%
# Dictionnaire des notes par série
notes_par_serie = {
    'SN': ["Sciences Naturelles", "Physiques Chimie", "Mathematiques", "Arabe", 
           "Français", "Anglais", "Instruction Réligieuse", "Education Physiques"],
    'M': ["Mathematiques", "Physiques Chimie", "Sciences Naturelles", "Arabe", 
          "Français", "Anglais", "Instruction Réligieuse", "Education Physique"],
    'LO': ["Droit Musulman", "ARABE", "Pensée Islamique", "Histoire-Géographie", 
           "Coran et Hadith", "Français", "Mathématiques", "Education Physique"],
    'TM': ["Construction Mecanique", "Mathématiques", "Sciences Physiques", "Français", 
           "Arabe", "Analyse de Fabrication", "Atelier", "Technologie_et_Automatisme"],
    'LM': ["Arabe", "Philosophie", "Français", "Histoire geographie", "Anglais", 
           "Mathématiques", "Instruction Réligieuse", "Education Physique"]
}

print("🚨 SIGNIFICATION DES NOTES PAR SÉRIE :")
print("=" * 40)

for serie, matieres in notes_par_serie.items():
    print(f"\n📚 SÉRIE {serie} :")
    for i, matiere in enumerate(matieres, 1):
        print(f"  Note{i} = {matiere}")

# %% [markdown]
"""
## 🎯 Explication Pédagogique : Pourquoi Exclure les Notes ?

### 🤔 Question Fréquente des Étudiants :
*"Mais les notes sont sûrement les meilleurs prédicteurs de réussite !"*

### ✅ Réponse Pédagogique :
**C'est vrai, MAIS...**

#### 🎯 **Objectif Réel :**
- Aider les enseignants à identifier les élèves à risque **DÈS SEPTEMBRE**
- Permettre des interventions pédagogiques **AVANT** qu'il ne soit trop tard
- Optimiser l'allocation des ressources éducatives **EN AMONT**

#### ⏰ **Problème Temporel :**
- Les notes du BAC sont connues en **JUIN**
- À ce moment, il est **TROP TARD** pour intervenir
- L'élève a déjà passé ses examens !

#### 💡 **Analogie Simple :**
C'est comme prédire la météo **après** qu'il ait plu !
"""

# %%
# Exemple concret pour les enseignants
print("💡 EXEMPLE CONCRET POUR LES ENSEIGNANTS :")
print("-" * 45)

exemple_scenarios = {
    'Scénario': ['Avec Notes (Inutile)', 'Sans Notes (Utile)'],
    'Moment': ['Juin - Après BAC', 'Septembre - Début année'],
    'Information': ['Notes connues', 'Profil élève connu'],
    'Prédiction': ['Réussite/Échec', 'Risque d\'échec'],
    'Action_Possible': ['Aucune (trop tard)', 'Soutien personnalisé'],
    'Impact': ['Zéro', 'Prévention efficace']
}

df_scenarios = pd.DataFrame(exemple_scenarios)
print(df_scenarios.to_string(index=False))

print("\n🎯 CONCLUSION :")
print("Utiliser les notes = Prédire le passé (inutile)")
print("Utiliser le profil = Prédire l'avenir (utile)")

# %% [markdown]
"""
## 🔍 Variables Clés pour Notre Modèle

### 🎯 Top 5 des Variables Prédictives
Basé sur l'expérience pédagogique et la littérature :
"""

# %%
# Variables les plus importantes
variables_importantes = {
    'Rang': [1, 2, 3, 4, 5],
    'Variable': ['Serie', 'Sexe', 'Age', 'Wilaya', 'Etablissement'],
    'Justification': [
        'Certaines filières sont plus difficiles',
        'Différences de performance historiques',
        'Âge optimal pour la réussite scolaire',
        'Disparités régionales en éducation',
        'Qualité variable des établissements'
    ],
    'Exemple_Impact': [
        'Série M plus exigeante que LO',
        'Les filles souvent plus assidues',
        'Redoublants plus à risque',
        'Nouakchott vs régions rurales',
        'Lycées privés vs publics'
    ]
}

df_importantes = pd.DataFrame(variables_importantes)
print("🏆 TOP 5 DES VARIABLES PRÉDICTIVES :")
print(df_importantes.to_string(index=False))

# %% [markdown]
"""
## 📊 Codage des Variables Catégorielles

### 🔤 Transformation pour l'IA
Les algorithmes ML ne comprennent que les nombres. Il faut transformer :
"""

# %%
# Exemples de codage
print("🔄 EXEMPLES DE CODAGE POUR L'IA :")
print("-" * 35)

# Exemple Sexe
print("👥 Variable SEXE :")
print("  Masculin → 0")
print("  Féminin  → 1")

print("\n📚 Variable SÉRIE :")
series_codes = {'SN': 0, 'M': 1, 'LO': 2, 'TM': 3, 'LM': 4}
for serie, code in series_codes.items():
    print(f"  {serie} → {code}")

print("\n🎯 Variable DECISION (Cible) :")
print("  Ajourné → 0 (Échec)")
print("  Admis OU Sessionnaire → 1 (Réussite)")

# %% [markdown]
"""
## 🎯 CORRECTION IMPORTANTE : Définition de la Réussite

### 🚨 Variable Cible Corrigée
**Notre objectif :** Prédire qui va **RÉUSSIR** vs **ÉCHOUER** au BAC

#### ✅ **Définition Correcte :**
- **RÉUSSITE (1) :** Admis OU Sessionnaire
  - **Admis :** Réussite directe (excellent !)
  - **Sessionnaire :** Réussite après rattrapage (bien !)
- **ÉCHEC (0) :** Ajourné
  - **Ajourné :** Échec total, doit redoubler

#### 🎯 **Logique Pédagogique :**
- **Admis ET Sessionnaire** = Obtention du BAC = SUCCÈS
- **Ajourné** = Pas de BAC = ÉCHEC
- **Exclus de l'analyse :** Absent, Examen Annulé (cas particuliers)

#### 💡 **Pourquoi Cette Logique ?**
Un élève **Sessionnaire** finit par obtenir son BAC après rattrapage.
C'est donc une **réussite**, même si elle nécessite plus d'efforts.

### 🎓 Impact pour les Enseignants
Cette prédiction permet d'identifier les élèves qui risquent de :
- **Redoubler** complètement (Ajourné)
- Versus ceux qui **obtiendront leur BAC** (Admis/Sessionnaire)
"""

# %%
print("🎯 DÉFINITION FINALE DE LA VARIABLE CIBLE :")
print("-" * 45)

definition_cible = {
    'Résultat_BAC': ['Admis', 'Sessionnaire', 'Ajourné', 'Absent', 'Examen Annulé'],
    'Code_IA': [1, 1, 0, 'Exclu', 'Exclu'],
    'Signification': [
        'Réussite directe',
        'Réussite après rattrapage', 
        'Échec - Redoublement',
        'Cas particulier',
        'Cas particulier'
    ],
    'Obtient_BAC': ['✅ Oui', '✅ Oui', '❌ Non', '❓ N/A', '❓ N/A']
}

df_cible = pd.DataFrame(definition_cible)
print("📊 CODAGE DE LA VARIABLE CIBLE :")
print(df_cible.to_string(index=False))

print(f"\n✅ RÉSUMÉ :")
print(f"  Succès (1) = Admis + Sessionnaire (obtiennent le BAC)")
print(f"  Échec (0) = Ajourné (redoublent)")
print(f"  Exclus = Absent + Examen Annulé (cas particuliers)")

# %% [markdown]
"""
## 🎓 Questions de Compréhension pour les Binômes

### ❓ Questions de Réflexion :

**Question 1 :** Pourquoi ne peut-on pas utiliser les notes pour prédire la réussite ?
- Réponse Binôme : ___________________________________

**Question 2 :** À quel moment de l'année scolaire notre modèle serait-il le plus utile ?
- Réponse Binôme : ___________________________________

**Question 3 :** Selon vous, quelle variable sera la plus prédictive ? Pourquoi ?
- Réponse Binôme : ___________________________________

**Question 4 :** Pourquoi considère-t-on "Sessionnaire" comme une réussite ?
- Réponse Binôme : ___________________________________

**Question 5 :** Comment un enseignant pourrait-il utiliser ces prédictions concrètement ?
- Réponse Binôme : ___________________________________

### 💡 Applications Pratiques :

**Scénario 1 :** Un élève de 19 ans, série M, région rurale
- Prédiction probable : ___________________________________
- Actions recommandées : ___________________________________

**Scénario 2 :** Une élève de 17 ans, série LO, Nouakchott, lycée privé
- Prédiction probable : ___________________________________
- Actions recommandées : ___________________________________
"""

# %%
print("✅ GUIDE DES VARIABLES TERMINÉ !")
print("=" * 35)
print("🎯 Vous comprenez maintenant :")
print("  ✓ Pourquoi exclure les notes")
print("  ✓ Signification de chaque variable")
print("  ✓ Variables les plus importantes")
print("  ✓ Comment coder pour l'IA")
print("  ✓ Définition correcte Réussite/Échec")
print("\n🚀 Prêts pour la classification !")

# %% [markdown]
"""
## 🔗 Connexion avec le Projet Principal

### 📋 Rappel du Contexte :
- **Projet :** Assistant IA pour l'Éducation Mauritanienne
- **Composante 1 :** Classification (prédire Réussite/Échec BAC)
- **Objectif :** Identifier les élèves à risque de redoublement dès septembre
- **Format :** Travail en binômes

### 🎯 Prochaine Étape :
Maintenant que vous comprenez les variables, vous allez :
1. **Charger** le dataset BAC 2022
2. **Explorer** les patterns dans les données
3. **Entraîner** 4 algorithmes de classification
4. **Interpréter** les résultats pour l'éducation

### 💪 Vous Êtes Prêts !
Avec cette compréhension des variables, vous pouvez maintenant créer 
un modèle IA qui aura un **impact réel** sur l'éducation en Mauritanie !
"""
