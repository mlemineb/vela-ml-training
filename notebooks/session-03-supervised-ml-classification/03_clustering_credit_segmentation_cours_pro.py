# %% [markdown]
"""
# Segmentation Clients (Crédit) – Parcours Complet et Métiers

But: produire des segments actionnables pour un Directeur Marketing/CRM en combinant
- Valeur client
- Risque/comportement de paiement
- Intensité d'usage du crédit

Nous allons:
1) Cadrer le besoin business et définir des KPI
2) Préparer des données orientées métier (features dérivées)
3) Standardiser, traiter les distributions et réduire la dimension pour visualiser
4) Appliquer 3 méthodes de clustering: K-Means, CAH, DBSCAN
5) Évaluer (Silhouette, Calinski-Harabasz, Davies-Bouldin) et tester la stabilité
6) Profiler et nommer les segments (personas)
7) Estimer un impact business simple (uplift) et recommander des actions
8) Exporter les artefacts (segments, profils)

Tout est exécuté automatiquement. Les widgets en fin de notebook permettent d'explorer.
"""

# %%
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("Environnement prêt")

# %% [markdown]
"""
## 1) Cadrage Business & KPI

Rôles et objectifs:
- Acquisition/Marketing: cibler des offres pertinentes, augmenter le revenu par client (ARPU)
- Risque/Recouvrement: réduire retards et défauts, améliorer DSO
- Fidélisation/CRM: réduire le churn, augmenter la durée de vie client (LTV)

KPI retenus pour juger nos segments:
- Couverture et taille de segment (% base)
- Valeur moyenne (LIMIT_BAL, montants facturés/payés)
- Risque: historique de retard (PAY_x), ratio de remboursement
- Potentiel d'upsell: taux d'utilisation du crédit
"""

# %% [markdown]
"""
## 2) Données & Préparation orientée métier
Dataset: "Default of Credit Card Clients" — variables socio-démo et financières.

Le chargement ci-dessous est robuste: il tente d'abord la source officielle UCI (Excel),
puis accepte un chemin local si fourni.
"""

# %%
# Charge le dataset depuis différentes sources avec fallback

def load_credit_data(local_path: str | None = None, sample_n: int | None = 1000) -> pd.DataFrame:
    """Charge le dataset 'Default of Credit Card Clients' avec plusieurs stratégies.
    - Si local_path est fourni et existe: lit CSV/XLS(X) local
    - Sinon tente l'Excel officiel UCI (XLS)
    - Optionnel: renvoie un échantillon (sample_n)
    """
    def _read_any(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.csv', '.txt']:
            return pd.read_csv(path)
        else:
            # Excel (xls/xlsx). L'original UCI est .xls
            try:
                return pd.read_excel(path, header=1, index_col=0)
            except Exception as e:
                # Dernier essai sans header/index spécifiques
                return pd.read_excel(path)

    # 1) Local si fourni
    if local_path and os.path.exists(local_path):
        df_local = _read_any(local_path)
        print(f"Données chargées depuis le fichier local: {local_path}")
        return df_local.sample(sample_n, random_state=42) if sample_n and len(df_local) > sample_n else df_local

    # 2) Source officielle UCI (XLS)
    uci_xls = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    try:
        df = pd.read_excel(uci_xls, header=1, index_col=0)
        print("Données chargées depuis UCI (Excel)")
        return df.sample(sample_n, random_state=42) if sample_n and len(df) > sample_n else df
    except Exception as e:
        print("Avertissement: Échec du chargement UCI Excel.")
        print("Conseil: si l'erreur concerne xlrd, installez-le: pip install xlrd")
        raise

# Utilisation du loader (adapter local_path si besoin)
try:
    df = load_credit_data(local_path=None, sample_n=1000)
except Exception as e:
    raise SystemExit(f"Impossible de charger le dataset. Spécifiez un chemin local (CSV/XLS). Erreur: {e}")

print("Dimensions:", df.shape)
# Aperçu
df.head(3)

# Sélection de variables brutes
base_cols = [
    'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
    'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'
]
base_cols = [c for c in base_cols if c in df.columns]
X_raw = df[base_cols].replace([np.inf,-np.inf], np.nan).dropna().copy()
print("Variables brutes:", len(base_cols))

# Feature engineering métier
fe = X_raw.copy()

# Utilisation du crédit (ex: facture récente / limite)
fe['utilization'] = np.clip(fe['BILL_AMT1'] / (fe['LIMIT_BAL']+1e-6), 0, 5)

# Ratio de remboursement récent
fe['repay_ratio_1'] = np.clip(fe['PAY_AMT1'] / (fe['BILL_AMT1']+1e-6), 0, 5)
fe['repay_ratio_mean'] = np.clip((fe[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis=1)) /
                                 (fe[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].sum(axis=1)+1e-6), 0, 5)

# Montants moyens
fe['bill_avg'] = fe[[c for c in fe.columns if c.startswith('BILL_AMT')]].mean(axis=1)
fe['pay_avg']  = fe[[c for c in fe.columns if c.startswith('PAY_AMT')]].mean(axis=1)

# Comptage de retards (PAY_x > 0)
pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
fe['delinq_count'] = (fe[pay_cols] > 0).sum(axis=1)
fe['recent_delinq'] = (fe['PAY_0'] > 0).astype(int)

# Tendance de facture (croissance récente)
fe['bill_trend'] = (fe['BILL_AMT1'] - fe['BILL_AMT3'])/(np.abs(fe['BILL_AMT3'])+1e-6)
fe['pay_trend']  = (fe['PAY_AMT1'] - fe['PAY_AMT3'])/(np.abs(fe['PAY_AMT3'])+1e-6)

# Log-transform sur variables fortement asymétriques
for col in ['LIMIT_BAL','bill_avg','pay_avg','BILL_AMT1','PAY_AMT1']:
    if col in fe.columns:
        fe[f'log_{col.lower()}'] = np.log1p(np.clip(fe[col], 0, None))

# Variables finales pour clustering
cluster_features = [
    'AGE','MARRIAGE','EDUCATION','SEX',
    'utilization','repay_ratio_1','repay_ratio_mean',
    'bill_avg','pay_avg','delinq_count','recent_delinq','bill_trend','pay_trend',
    'log_limit_bal','log_bill_amt1','log_pay_amt1'
]
cluster_features = [c for c in cluster_features if c in fe.columns]
X_model = fe[cluster_features].copy()
print("Variables de clustering:", cluster_features)
print("Dimensions prêtes:", X_model.shape)

# %% [markdown]
"""
## 3) Standardisation, réduction de dimension, contrôle des distributions
"""

# %%
scaler = StandardScaler()
X = scaler.fit_transform(X_model)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
print(f"Variance expliquée (2D): {pca.explained_variance_ratio_.sum():.1%}")

plt.scatter(X_2d[:,0], X_2d[:,1], s=8, alpha=0.5)
plt.title('Projection PCA (pré-visualisation)')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.grid(True, alpha=0.2)
plt.show()

# t-SNE (option visuelle complémentaire)
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], s=8, alpha=0.5)
plt.title('Projection t-SNE (non linéaire)')
plt.xlabel('Dim 1'); plt.ylabel('Dim 2'); plt.grid(True, alpha=0.2)
plt.show()

# %%
# Outils

def plot_clusters(X2, labels, title="Clusters", centers_2d=None):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    scatter = plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='viridis', s=10, alpha=0.7)
    plt.title(title); plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04); cbar.set_label('Cluster')
    if centers_2d is not None:
        plt.scatter(centers_2d[:,0], centers_2d[:,1], c='red', marker='x', s=150, linewidths=3, label='Centres')
        plt.legend()

    plt.subplot(1,2,2)
    unique, counts = np.unique(labels, return_counts=True)
    colors = ['#bbbbbb' if u == -1 else None for u in unique]
    plt.bar([str(u) for u in unique], counts, color=colors)
    plt.title('Taille des segments'); plt.xlabel('Cluster (-1=bruit)'); plt.ylabel('Effectif')
    plt.tight_layout(); plt.show()


def metric_report(X, labels):
    labs = set(labels)
    if len(labs) <= 1 or (len(labs)==2 and -1 in labs):
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski": calinski_harabasz_score(X, labels),
        "davies": davies_bouldin_score(X, labels)
    }

# %% [markdown]
"""
## 4) Clustering – K-Means, CAH, DBSCAN
"""

# %%
# K-Means
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=20)
labels_km = kmeans.fit_predict(X)
centers_2d = PCA(n_components=2, random_state=42).fit_transform(kmeans.cluster_centers_)
rep_km = metric_report(X, labels_km)
print("K-Means (K=5)", rep_km)
plot_clusters(X_2d, labels_km, title=f"K-Means (PCA) | Sil={rep_km['silhouette']:.3f}", centers_2d=centers_2d)
plot_clusters(X_tsne, labels_km, title=f"K-Means (t-SNE) | Sil={rep_km['silhouette']:.3f}")

# CAH (Ward)
ag = AgglomerativeClustering(n_clusters=K, linkage='ward')
labels_cah = ag.fit_predict(X)
rep_cah = metric_report(X, labels_cah)
print("CAH (Ward, K=5)", rep_cah)
plot_clusters(X_2d, labels_cah, title=f"CAH-Ward (PCA) | Sil={rep_cah['silhouette']:.3f}")

# DBSCAN – eps heuristique via k-dist (k=5)
nbrs = NearestNeighbors(n_neighbors=5).fit(X)
distances, _ = nbrs.kneighbors(X)
kdist = np.sort(distances[:, -1])
EPS = float(np.percentile(kdist, 90)); MIN_SAMPLES = 10

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
labels_db = db.fit_predict(X)
rep_db = metric_report(X, labels_db)
noise = np.mean(labels_db==-1)
print(f"DBSCAN (eps={EPS:.2f}, min={MIN_SAMPLES})", rep_db, f"| bruit={noise:.1%}")
plot_clusters(X_2d, labels_db, title=f"DBSCAN (PCA) | Sil={rep_db['silhouette'] if not np.isnan(rep_db['silhouette']) else 'NA'} | bruit={noise:.1%}")

# %% [markdown]
"""
## 5) Robustesse/Stabilité (idée simple)
- K-Means est relancé sur des sous-échantillons bootstrap, on mesure l'ARI pair-à-pair
- Une ARI moyenne élevée ~ segments stables
"""

# %%
aris = []
base_labels = labels_km
for seed in range(5):
    X_boot, idx = resample(X, np.arange(len(X)), replace=True, n_samples=int(0.8*len(X)), random_state=seed)
    km_b = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X_boot)
    # Projection des centroïdes sur l'ensemble complet pour prédire des labels approximatifs
    # (assignation par plus proche centre)
    centers = km_b.cluster_centers_
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels_projected = dists.argmin(axis=1)
    aris.append(adjusted_rand_score(base_labels, labels_projected))

print("Stabilité K-Means – ARI sur 5 bootstrap:", [f"{a:.3f}" for a in aris], "| Moy=", f"{np.mean(aris):.3f}")

# %% [markdown]
"""
## 6) Profiling riche et Nommage des segments (personas)
"""

# %%
# Choix du modèle final (ici K-Means pour lisibilité)
labels_final = labels_km
segments = fe.copy()
segments['cluster'] = labels_final

# Indicateurs par segment
profile_cols = [
    'AGE','utilization','repay_ratio_1','repay_ratio_mean','delinq_count','recent_delinq',
    'bill_avg','pay_avg','bill_trend','pay_trend','LIMIT_BAL'
]
profile_cols = [c for c in profile_cols if c in segments.columns]
segment_profile = segments.groupby('cluster')[profile_cols].agg(['mean','median','count']).round(2)
segment_profile

# Nommage simple (exemple, à adapter selon vos observations)
personas = {}
for cid, grp in segments.groupby('cluster'):
    util = grp['utilization'].mean() if 'utilization' in grp else 0
    risk = grp['delinq_count'].mean() if 'delinq_count' in grp else 0
    repay = grp['repay_ratio_mean'].mean() if 'repay_ratio_mean' in grp else 0
    limit = grp['LIMIT_BAL'].mean() if 'LIMIT_BAL' in grp else 0

    if util > 0.6 and repay > 0.8 and risk < 0.5:
        name = "Premium forts utilisateurs, bons payeurs"
    elif risk >= 1.0 and repay < 0.6:
        name = "Risque élevé, besoin d'accompagnement"
    elif util < 0.2 and limit < segments['LIMIT_BAL'].median():
        name = "Sous-utilisateurs à faible potentiel"
    else:
        name = "Équilibrés / généralistes"
    personas[cid] = name

pd.DataFrame.from_dict(personas, orient='index', columns=['Persona'])

# Visualisation comparative
fig, axes = plt.subplots(1, 3, figsize=(16,4))
for ax, col in zip(axes, ['utilization','repay_ratio_mean','delinq_count']):
    if col in segments:
        sns.boxplot(data=segments, x='cluster', y=col, ax=ax)
        ax.set_title(col)
plt.tight_layout(); plt.show()

# %% [markdown]
"""
## 7) Synthèse rapide avant le cas métier

Ce que nous avons couvert jusqu'ici:

- Données et préparation
  - Chargement robuste (UCI Excel ou fichier local)
  - Sélection de variables financières/socio-démo
  - Création de variables à forte valeur métier: `utilization`, `repay_ratio_*`, `delinq_count`, tendances `bill_trend`/`pay_trend`, logs

- Pré-traitement et visualisation
  - Standardisation (éviter la domination d'une variable)
  - Réduction de dimension (PCA 2D pour vue globale, t-SNE pour structures non linéaires)

- Algorithmes de clustering
  - K-Means: rapide, lisible, clusters compacts; nécessite K
  - CAH (Ward): hiérarchie de regroupements, interprétation par dendrogramme; nécessite K
  - DBSCAN: pas de K, gère bruit/formes variées; sensible à `eps`/`min_samples`

- Évaluation et robustesse
  - Métriques internes: Silhouette (↑ mieux), Calinski-Harabasz (↑), Davies-Bouldin (↓)
  - Stabilité (bootstrap + ARI): segments robustes vs sensibles à l’échantillonnage

- Choix du modèle
  - Combiner: qualité (métriques), lisibilité, stabilité et actionnabilité métier

Nous allons maintenant passer à l’estimation d’impact et aux recommandations par segment.
"""

# %% [markdown]
"""
## 8) Esquisse d'Impact Business (estimation simple)
Hypothèses illustratives (à adapter):
- Sur le segment "Premium": +10% de limite -> +5% d'utilisation -> +2% de revenu net
- Segment "Risque élevé": plan de régularisation -> -20% retards -> -1% pertes
- Segment "Sous-utilisateurs": campagne réactivation -> +10% utilisation -> +1% revenu net

On évalue un potentiel relatif (indicatif), non une P&L réelle.
"""

# %%
# Estimation jouet de gains par segment
estimates = []
for cid, grp in segments.groupby('cluster'):
    size = len(grp)
    util = grp['utilization'].mean() if 'utilization' in grp else 0
    limit = grp['LIMIT_BAL'].mean() if 'LIMIT_BAL' in grp else 0
    risk = grp['delinq_count'].mean() if 'delinq_count' in grp else 0

    persona = personas[cid]
    if "Premium" in persona:
        gain = 0.02 * size  # proxy
    elif "Risque" in persona:
        gain = 0.01 * size  # économie pertes
    elif "Sous-utilisateurs" in persona:
        gain = 0.01 * size
    else:
        gain = 0.005 * size
    estimates.append({"cluster": cid, "persona": persona, "taille": size, "gain_relatif": round(gain,2)})

est_df = pd.DataFrame(estimates).sort_values('gain_relatif', ascending=False)
est_df

plt.bar(est_df['cluster'].astype(str), est_df['gain_relatif'])
plt.title("Hiérarchisation indicative de l'impact (proxy)")
plt.xlabel('Cluster'); plt.ylabel('Gain relatif (unité arbitraire)'); plt.grid(True, axis='y', alpha=0.3)
plt.show()

# %% [markdown]
"""
## 9) Export des Artefacts
"""

# %%
# Export des labels et profils
export = df.loc[X_raw.index].copy()
export['cluster'] = labels_final
export.to_csv('credit_segments_labels.csv', index=False)
segment_profile.to_csv('credit_segments_profile.csv')
print("Exports: credit_segments_labels.csv, credit_segments_profile.csv")

# %% [markdown]
"""
## 10) Exploration interactive (facultatif)
"""

# %%
@interact(k=IntSlider(min=2, max=10, step=1, value=K, description='K (K-Means)'))
def explore_kmeans(k):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    lab = km.fit_predict(X)
    rep = metric_report(X, lab)
    ctr2 = PCA(n_components=2, random_state=42).fit_transform(km.cluster_centers_)
    plot_clusters(X_2d, lab, title=f"K-Means (PCA) K={k} | Sil={rep['silhouette'] if not np.isnan(rep['silhouette']) else 'NA'}", centers_2d=ctr2)

@interact(k=IntSlider(min=2, max=10, step=1, value=K, description='K (CAH)'),
          link=Dropdown(options=['ward','complete','average','single'], value='ward', description='Linkage'))
def explore_cah(k, link):
    ag = AgglomerativeClustering(n_clusters=k, linkage=link)
    lab = ag.fit_predict(X)
    rep = metric_report(X, lab)
    plot_clusters(X_2d, lab, title=f"CAH-{link} (PCA) K={k} | Sil={rep['silhouette'] if not np.isnan(rep['silhouette']) else 'NA'}")

# DBSCAN helpers
kth = np.sort(NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)[0][:, -1])
@interact(eps=FloatSlider(min=float(np.percentile(kth, 50)), max=float(np.percentile(kth, 99)), step=0.05, value=float(np.percentile(kth, 90)), description='eps'),
          ms=IntSlider(min=3, max=30, step=1, value=10, description='min_samples'))
def explore_dbscan(eps, ms):
    db = DBSCAN(eps=eps, min_samples=ms)
    lab = db.fit_predict(X)
    rep = metric_report(X, lab)
    noise = np.mean(lab==-1)
    plot_clusters(X_2d, lab, title=f"DBSCAN (PCA) eps={eps:.2f}, min={ms} | Sil={rep['silhouette'] if not np.isnan(rep['silhouette']) else 'NA'} | Bruit={noise:.1%}")

# %% [markdown]
"""
## 11) Conclusion & Recos
- Les segments sont définis avec des variables métier (utilisation, remboursement, risque)
- Évaluation multi-métriques + stabilité
- Personas nommés et hiérarchisation d'impact

Prochaines étapes:
- Valider avec les équipes Risque/Marketing (sens business)
- Tester des offres par segment (A/B test) et mesurer KPI
- Itérer sur le feature engineering (revenus, ancienneté, canaux, interactions)
- Enrichir via d'autres algorithmes (GMM, Spectral, HDBSCAN)
"""
