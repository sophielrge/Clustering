import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

dataset = "banana"

# ---- Création dossier de sortie ----
output_dir = f"src/assets/agglomerative/{dataset}"
os.makedirs(output_dir, exist_ok=True)

# ---- Lire fichier ARFF ----
def lire_fichier_arff(chemin_fichier):
    points = []
    with open(chemin_fichier, 'r') as f:
        data_started = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper() == '@DATA':
                data_started = True
                continue
            if data_started:
                valeurs = line.split(',')
                points.append([float(valeurs[0]), float(valeurs[1])])
    return np.array(points)

# ---- Tester différents n_clusters ----
def tester_n_clusters(points, linkage='ward', max_clusters=10):
    silhouettes = []
    calinskis = []
    davies = []
    cluster_range = range(2, max_clusters+1)

    for k in cluster_range:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(points)

        silhouettes.append(silhouette_score(points, labels))
        calinskis.append(calinski_harabasz_score(points, labels))
        davies.append(davies_bouldin_score(points, labels))

    # Graphiques des scores
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.plot(cluster_range, silhouettes, 'bo-')
    plt.xlabel("n_clusters"); plt.ylabel("Score"); plt.title(f"Silhouette ({linkage})")
    plt.grid(alpha=0.3)

    plt.subplot(2,2,2)
    plt.plot(cluster_range, calinskis, 'go-')
    plt.xlabel("n_clusters"); plt.ylabel("Score"); plt.title(f"Calinski-Harabasz ({linkage})")
    plt.grid(alpha=0.3)

    plt.subplot(2,2,3)
    plt.plot(cluster_range, davies, 'ro-')
    plt.xlabel("n_clusters"); plt.ylabel("Score"); plt.title(f"Davies-Bouldin ({linkage}) (bas = mieux)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/scores_n_clusters_{linkage}.png")
    plt.close()

    meilleur_k = cluster_range[np.argmax(silhouettes)]
    print(f"[{linkage}] Meilleur n_clusters (Silhouette) : {meilleur_k}")
    return meilleur_k

# ---- Dendrogramme ----
def afficher_dendrogramme(points, method='ward'):
    linked = scipy_linkage(points, method=method)
    plt.figure(figsize=(10,5))
    dendrogram(linked, truncate_mode='level', p=30)
    plt.title(f"Dendrogramme ({method})")
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dendrogram_{method}.png")
    plt.close()

# ---- Visualisation clusters ----
def visualiser_clusters(points, n_clusters, linkage):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(points)

    plt.figure(figsize=(7,5))
    plt.scatter(points[:,0], points[:,1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f"Agglomerative Clustering ({linkage}, k={n_clusters})")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(alpha=0.3)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/clusters_{linkage}_k{n_clusters}.png")
    plt.close()

# ---- Programme principal ----
def main():
    chemin_fichier = f"src/dataset/artificial/{dataset}.arff"
    points = lire_fichier_arff(chemin_fichier)
    print(f"{len(points)} points chargés dans {dataset}")

    linkages = ['ward', 'single', 'complete', 'average']
    best_k_for_linkage = {}
    silhouette_scores = []

    # ---- Traiter CHAQUE linkage ----
    for link in linkages:
        print("\n==============================")
        print(f"   TEST LINKAGE = {link}")
        print("==============================")

        # 1) choix du meilleur k
        k_best = tester_n_clusters(points, linkage=link, max_clusters=10)
        best_k_for_linkage[link] = k_best

        # 2) dendrogramme pour ce linkage
        afficher_dendrogramme(points, method=link)

        # 3) clusters obtenus pour ce linkage
        visualiser_clusters(points, n_clusters=k_best, linkage=link)

        # 4) score silhouette du résultat final pour comparaison
        model = AgglomerativeClustering(n_clusters=k_best, linkage=link)
        labels = model.fit_predict(points)
        silhouette_scores.append(silhouette_score(points, labels))

    # ---- Choix final du meilleur linkage ----
    best_linkage = linkages[np.argmax(silhouette_scores)]
    best_k = best_k_for_linkage[best_linkage]

    print("\n====================================")
    print(f"Meilleur linkage global : {best_linkage}")
    print(f"Meilleur k : {best_k}")
    print("====================================")

    # ---- Visualisation finale ----
    visualiser_clusters(points, n_clusters=best_k, linkage=best_linkage)

if __name__ == "__main__":
    main()
