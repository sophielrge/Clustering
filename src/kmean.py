import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


dataset = "square4"

# ---- Création du dossier de sortie ----
output_dir = f"src/assets/kmean/{dataset}"
os.makedirs(output_dir, exist_ok=True)

def lire_fichier_arff(chemin_fichier):
    points = []
    
    with open(chemin_fichier, 'r') as fichier:
        lire_donnees = False
        
        for ligne in fichier:
            ligne = ligne.strip()
            
            if not ligne or ligne.startswith('%'):
                continue
            
            if ligne.upper() == '@DATA':
                lire_donnees = True
                continue
            
            if lire_donnees:
                valeurs = ligne.split(',')
                x = float(valeurs[0])
                y = float(valeurs[1])
                points.append([x, y])
    
    return np.array(points)

def tester_hyperparametre(points, parametre, valeurs, config_base=None):
    if config_base is None:
        config_base = {'n_clusters': 3, 'init': 'k-means++', 'max_iter': 300, 'n_init': 1, 'random_state': None}
    
    print(f"\nTEST {parametre}")
    print("=" * 50)
    
    # Stocker les résultats pour les graphiques
    silhouettes = []
    calinskis = []
    davies = []
    iterations = []
    scores_combinés = []
    
    meilleur_score = -1
    meilleure_valeur = valeurs[0]
    
    for valeur in valeurs:
        config = config_base.copy()
        config[parametre] = valeur
        
        # Appliquer K-means
        kmeans = KMeans(**config)
        labels = kmeans.fit_predict(points)

        silhouette = silhouette_score(points, labels)
        calinski = calinski_harabasz_score(points, labels)
        davies_score = davies_bouldin_score(points, labels)

        silhouettes.append(silhouette)
        calinskis.append(calinski)
        davies.append(davies_score)
        iterations.append(kmeans.n_iter_)

        score_combiné = (0.6 * silhouette) + (0.3 * (calinski / 1000)) + (0.1 * (1 / davies_score))
        scores_combinés.append(score_combiné)

        if score_combiné > meilleur_score:
            meilleur_score = score_combiné
            meilleure_valeur = valeur

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Analyse des performances pour {parametre}', fontsize=16, fontweight='bold')
    
    # Graphique 1: Score de silhouette
    axes[0, 0].plot(valeurs, silhouettes, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel(parametre)
    axes[0, 0].set_ylabel('Score Silhouette')
    axes[0, 0].set_title('Score Silhouette vs ' + parametre)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(meilleure_valeur, color='red', linestyle='--', alpha=0.7, label=f'Meilleur: {meilleure_valeur}')
    axes[0, 0].legend()
    
    # Graphique 2: Score Calinski-Harabasz
    axes[0, 1].plot(valeurs, calinskis, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel(parametre)
    axes[0, 1].set_ylabel('Score Calinski-Harabasz')
    axes[0, 1].set_title('Score Calinski-Harabasz vs ' + parametre)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(meilleure_valeur, color='red', linestyle='--', alpha=0.7, label=f'Meilleur: {meilleure_valeur}')
    axes[0, 1].legend()
    
    # Graphique 3: Score Davies-Bouldin (plus bas = mieux)
    axes[1, 0].plot(valeurs, davies, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel(parametre)
    axes[1, 0].set_ylabel('Score Davies-Bouldin')
    axes[1, 0].set_title('Score Davies-Bouldin vs ' + parametre + '\n(Plus bas = mieux)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(meilleure_valeur, color='red', linestyle='--', alpha=0.7, label=f'Meilleur: {meilleure_valeur}')
    axes[1, 0].legend()
    
    # Graphique 4: Score combiné ou nombre d'itérations
    if parametre == 'max_iter':
        axes[1, 1].plot(valeurs, iterations, 'mo-', linewidth=2, markersize=8, label='Itérations réelles')
        axes[1, 1].set_xlabel('max_iter')
        axes[1, 1].set_ylabel('Itérations réelles')
        axes[1, 1].set_title('Itérations réelles vs max_iter')
    else:
        axes[1, 1].plot(valeurs, scores_combinés, 'co-', linewidth=2, markersize=8, label='Score combiné')
        axes[1, 1].set_xlabel(parametre)
        axes[1, 1].set_ylabel('Score combiné')
        axes[1, 1].set_title('Score combiné vs ' + parametre)
    
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(meilleure_valeur, color='red', linestyle='--', alpha=0.7, label=f'Meilleur: {meilleure_valeur}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'src/assets/kmean/{dataset}/test_{parametre}_{dataset}.png')
    plt.close()
    
    print(f"MEILLEUR {parametre}: {meilleure_valeur}")
    return meilleure_valeur

def comparer_init(points, n_clusters=3, max_iter=300, n_init=1):
    inits = ['k-means++', 'random']
    resultats = []

    for init_method in inits:
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter, n_init=n_init)
        labels = kmeans.fit_predict(points)
        
        silhouette = silhouette_score(points, labels)
        calinski = calinski_harabasz_score(points, labels)
        davies = davies_bouldin_score(points, labels)
        score_combine = (0.6 * silhouette) + (0.3 * (calinski / 1000)) + (0.1 * (1 / davies))
        
        resultats.append({
            'Init': init_method,
            'Silhouette': round(silhouette, 3),
            'Calinski-Harabasz': round(calinski, 3),
            'Davies-Bouldin': round(davies, 3),
            'Score combiné': round(score_combine, 3)
        })
    
    df = pd.DataFrame(resultats)
    print("\nComparaison des méthodes d'initialisation")
    print(df.to_markdown(f"src/assets/kmean/{dataset}/comparaison_init_{dataset}.md", index=False))
    return df

def visualiser_clusters_finaux(points, config_optimale):
    kmeans = KMeans(**config_optimale)
    labels = kmeans.fit_predict(points)
    centres = kmeans.cluster_centers_
    
    plt.figure(figsize=(12, 5))
    
    # Graphique des clusters
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    plt.scatter(centres[:, 0], centres[:, 1], c='red', marker='X', s=200, linewidths=2, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(f'Clusters K-means (K={config_optimale["n_clusters"]})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'src/assets/kmean/{dataset}/clusters_finaux_{dataset}.png')
    plt.close()

def tester_hyperparametres_complet(chemin_fichier):
    points = lire_fichier_arff(chemin_fichier)
    print(f"{len(points)} points chargés")
    print("n_init=1 pour voir les vraies différences entre les méthodes")
    
    # Configuration de base avec n_init=1
    config = {'n_init': 10, 'random_state': 42}

    meilleur_k = tester_hyperparametre(
        points, 
        'n_clusters', 
        [2,3,4,5,6,7,8,9],
        config
    )
    config['n_clusters'] = meilleur_k

    comparer_init(points, n_clusters=meilleur_k, max_iter=300, n_init=1)

    meilleur_max_iter = tester_hyperparametre(
        points,
        'max_iter',
        [1, 2, 5, 10, 20, 30, 40, 50, 100],
        config
    )
    config['max_iter'] = meilleur_max_iter

    resume_md = []
    resume_md.append(" _CONFIGURATION OPTIMALE_\n")

    for param, valeur in config.items():
        ligne = f"- _{param}_ : {valeur}"
        print(ligne)
        resume_md.append(ligne)

    with open(f"src/assets/kmean/{dataset}/config_optimale_{dataset}.md", "w") as f:
        f.write("\n".join(resume_md))

    visualiser_clusters_finaux(points, config)

if __name__ == "__main__":
    tester_hyperparametres_complet(f"src/dataset/artificial/{dataset}.arff")