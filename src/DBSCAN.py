import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

def lire_fichier_arff(chemin_fichier):
    """Lit un fichier ARFF et retourne les points"""
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

def trouver_epsilon_optimal(points, k=4):
    """
    Trouve la valeur epsilon optimale en utilisant la méthode du coude
    sur les distances aux k-plus proches voisins
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(points)
    distances, indices = neighbors_fit.kneighbors(points)
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points triés')
    plt.ylabel(f'Distance au {k}ème plus proche voisin')
    plt.title('Méthode du coude pour trouver epsilon optimal')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return distances

def tester_hyperparametre(points, parametre, valeurs, config_base=None):
    """
    Teste un hyperparamètre avec différentes valeurs et génère des graphiques
    """
    if config_base is None:
        config_base = {'eps': 0.5, 'min_samples': 5}
    
    print(f"\n🧪 TEST {parametre}")
    print("=" * 50)
    
    # Stocker les résultats pour les graphiques
    silhouettes = []
    calinskis = []
    davies = []
    n_clusters_list = []
    n_noise_list = []
    
    meilleur_score = -1
    meilleure_valeur = valeurs[0]
    
    for valeur in valeurs:
        # Créer la configuration avec la valeur testée
        config = config_base.copy()
        config[parametre] = valeur
        
        # Appliquer DBSCAN
        dbscan = DBSCAN(**config)
        labels = dbscan.fit_predict(points)
        
        # Compter les clusters (en ignorant le bruit -1)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        # Stocker les informations sur les clusters
        n_clusters_list.append(n_clusters)
        n_noise_list.append(n_noise)
        
        # Calculer les métriques seulement s'il y a au moins 2 clusters
        if n_clusters > 1:
            silhouette = silhouette_score(points, labels)
            calinski = calinski_harabasz_score(points, labels)
            davies_score = davies_bouldin_score(points, labels)
        else:
            # Valeurs par défaut si pas assez de clusters
            silhouette = -1
            calinski = 0
            davies_score = 10  # Valeur haute pour Davies (pire score)
        
        # Stocker les résultats
        silhouettes.append(silhouette)
        calinskis.append(calinski)
        davies.append(davies_score)
        
        # Score combiné (pénalisé si pas assez de clusters)
        if n_clusters > 1:
            score_combiné = (0.5 * silhouette) + (0.3 * (calinski / 1000)) + (0.2 * (1 / davies_score))
            # Pénalité pour le bruit
            score_combiné *= (1 - (n_noise / len(points)) * 0.3)
        else:
            score_combiné = -10  # Forte pénalité si pas de clusters
        
        if score_combiné > meilleur_score:
            meilleur_score = score_combiné
            meilleure_valeur = valeur
    
    # Créer un seul graphique avec les 4 courbes normalisées
    plt.figure(figsize=(12, 8))
    
    # Normaliser les données entre 0 et 1
    def normaliser(data):
        data = np.array(data)
        if np.max(data) == np.min(data):
            return np.ones_like(data) * 0.5
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Pour Silhouette : déjà entre -1 et 1, on ramène à 0-1
    silhouettes_norm = normaliser(silhouettes)
    
    # Pour Calinski : valeurs peuvent être grandes, on normalise
    calinskis_norm = normaliser(calinskis)
    
    # Pour Davies : plus bas = mieux, donc on prend l'inverse puis on normalise
    davies_inverse = [1/d if d > 0 else 0 for d in davies]
    davies_norm = normaliser(davies_inverse)
    
    # Pour Bruit : moins de bruit = mieux, donc on prend l'inverse puis on normalise
    bruit_inverse = [-n for n in n_noise_list]  # Négatif car moins de bruit = mieux
    bruit_norm = normaliser(bruit_inverse)
    
    # Tracer les 4 courbes normalisées
    plt.plot(valeurs, silhouettes_norm, 'bo-', linewidth=2, markersize=6, label='Silhouette (↑ mieux)')
    plt.plot(valeurs, calinskis_norm, 'go-', linewidth=2, markersize=6, label='Calinski (↑ mieux)')
    plt.plot(valeurs, davies_norm, 'ro-', linewidth=2, markersize=6, label='Davies-Inverse (↑ mieux)')
    plt.plot(valeurs, bruit_norm, 'mo-', linewidth=2, markersize=6, label='Bruit-Inverse (↑ mieux)')
    
    # Marquer le meilleur paramètre
    plt.axvline(meilleure_valeur, color='red', linestyle='--', alpha=0.8, 
                label=f'Meilleur {parametre}: {meilleure_valeur:.3f}')
    
    plt.xlabel(parametre)
    plt.ylabel('Scores normalisés (0-1)')
    plt.title(f'Analyse des performances pour {parametre}\n(Toutes métriques normalisées - ↑ = mieux)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Afficher aussi un graphique avec les valeurs réelles du bruit et nombre de clusters
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.gca()
    line1 = ax1.plot(valeurs, n_clusters_list, 'co-', linewidth=2, markersize=6, label='Nombre de clusters')
    ax1.set_xlabel(parametre)
    ax1.set_ylabel('Nombre de clusters', color='c')
    ax1.tick_params(axis='y', labelcolor='c')
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(valeurs, n_noise_list, 'mo-', linewidth=2, markersize=6, label='Points de bruit')
    ax2.set_ylabel('Nombre de points de bruit', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    
    plt.axvline(meilleure_valeur, color='red', linestyle='--', alpha=0.8, 
                label=f'Meilleur {parametre}: {meilleure_valeur:.3f}')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title(f'Clusters et Bruit vs {parametre}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 MEILLEUR {parametre}: {meilleure_valeur}")
    valeurs_list = list(valeurs)
    index_meilleur = valeurs_list.index(meilleure_valeur)
    print(f"   → Clusters: {n_clusters_list[index_meilleur]}, Bruit: {n_noise_list[index_meilleur]}")
    print(f"   → Silhouette: {silhouettes[index_meilleur]:.3f}, Calinski: {calinskis[index_meilleur]:.1f}, Davies: {davies[index_meilleur]:.3f}")
    
    return meilleure_valeur

def visualiser_clusters_finaux(points, config_optimale):
    """Visualise les clusters finaux avec la configuration optimale"""
    dbscan = DBSCAN(**config_optimale)
    labels = dbscan.fit_predict(points)
    
    # Compter les clusters et le bruit
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    plt.figure(figsize=(15, 5))
    
    # Graphique des clusters
    plt.subplot(1, 3, 1)
    
    # Couleurs pour les clusters, noir pour le bruit
    colors = [plt.cm.nipy_spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            label = 'Bruit'
            size = 20
            alpha = 0.6
        else:
            label = f'Cluster {k}'
            size = 50
            alpha = 0.7
        
        class_member_mask = (labels == k)
        xy = points[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], s=size, alpha=alpha, label=label, edgecolors='black' if k != -1 else None, linewidth=0.5)
    
    plt.title(f'Clusters DBSCAN\n(eps={config_optimale["eps"]}, min_samples={config_optimale["min_samples"]})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique des métriques
    plt.subplot(1, 3, 2)
    if n_clusters > 1:
        metriques = ['Silhouette', 'Calinski', 'Davies']
        scores = [
            silhouette_score(points, labels),
            calinski_harabasz_score(points, labels),
            davies_bouldin_score(points, labels)
        ]
        
        bars = plt.bar(metriques, scores, color=['blue', 'green', 'red'], alpha=0.7)
        plt.title('Scores finaux')
        plt.ylabel('Score')
        
        # Ajouter les valeurs sur les barres
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'Pas assez de clusters\npour calculer les métriques', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Métriques non disponibles')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # Graphique d'information
    plt.subplot(1, 3, 3)
    info_text = f"""
    RÉSULTATS DBSCAN
    
    Clusters trouvés: {n_clusters}
    Points de bruit: {n_noise}
    Total points: {len(points)}
    
    Paramètres:
    eps: {config_optimale['eps']}
    min_samples: {config_optimale['min_samples']}
    
    Bruit: {n_noise/len(points)*100:.1f}%
    """
    plt.text(0.1, 0.9, info_text, fontfamily='monospace', fontsize=10, verticalalignment='top')
    plt.axis('off')
    plt.title('Informations du clustering')
    
    plt.tight_layout()
    plt.show()

def tester_hyperparametres_complet(chemin_fichier):
    """Test complet avec visualisations graphiques pour DBSCAN"""
    
    # Charger les points
    points = lire_fichier_arff(chemin_fichier)
    print(f"✅ {len(points)} points chargés")
    
    # Trouver epsilon optimal avec la méthode du coude
    print("\n📊 Recherche de epsilon optimal avec méthode du coude...")
    distances = trouver_epsilon_optimal(points, k=4)
    
    # Suggérer une plage pour epsilon basée sur la courbe
    epsilon_suggestion = np.percentile(distances, 70)  # 70ème percentile comme point de départ
    print(f"💡 Epsilon suggéré: {epsilon_suggestion:.3f}")
    
    # Configuration de base
    config = {'eps': epsilon_suggestion, 'min_samples': 5}
    
    # 1. Test eps
    epsilons = np.linspace(epsilon_suggestion * 0.1, epsilon_suggestion * 3, 15)
    meilleur_eps = tester_hyperparametre(
        points, 
        'eps', 
        epsilons,
        config
    )
    config['eps'] = meilleur_eps
    
    # 2. Test min_samples
    meilleur_min_samples = tester_hyperparametre(
        points,
        'min_samples',
        [2, 3, 4, 5, 7, 10, 15, 20],
        config
    )
    config['min_samples'] = meilleur_min_samples
    
    # Résumé final et visualisation
    print("\n" + "=" * 50)
    print("🎯 CONFIGURATION OPTIMALE DBSCAN")
    print("=" * 50)
    for param, valeur in config.items():
        print(f"{param}: {valeur}")
    
    # Visualisation finale des clusters
    print("\n📈 VISUALISATION DES CLUSTERS FINAUX")
    visualiser_clusters_finaux(points, config)

# Test dans le main
if __name__ == "__main__":
    tester_hyperparametres_complet("/home/boaglio/5A/clustering/Clustering/src/dataset/artificial/2d-4c-no4.arff")