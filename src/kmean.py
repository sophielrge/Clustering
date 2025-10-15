import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

def tester_hyperparametre(points, parametre, valeurs, config_base=None):
    """
    Teste un hyperparamètre avec différentes valeurs et génère des graphiques
    """
    if config_base is None:
        config_base = {'n_clusters': 3, 'init': 'k-means++', 'max_iter': 300, 'n_init': 1}
    
    print(f"\n🧪 TEST {parametre}")
    print("=" * 50)
    
    # Stocker les résultats pour les graphiques
    silhouettes = []
    calinskis = []
    davies = []
    iterations = []
    
    meilleur_score = -1
    meilleure_valeur = valeurs[0]
    
    for valeur in valeurs:
        # Créer la configuration avec la valeur testée
        config = config_base.copy()
        config[parametre] = valeur
        
        # Appliquer K-means
        kmeans = KMeans(**config)
        labels = kmeans.fit_predict(points)
        
        # Calculer les métriques
        silhouette = silhouette_score(points, labels)
        calinski = calinski_harabasz_score(points, labels)
        davies_score = davies_bouldin_score(points, labels)
        
        # Stocker les résultats
        silhouettes.append(silhouette)
        calinskis.append(calinski)
        davies.append(davies_score)
        iterations.append(kmeans.n_iter_ if parametre == 'max_iter' else None)
        
        # Score combiné
        score_combiné = (0.6 * silhouette) + (0.3 * (calinski / 1000)) + (0.1 * (1 / davies_score))
        
        if score_combiné > meilleur_score:
            meilleur_score = score_combiné
            meilleure_valeur = valeur
    
    # Créer les graphiques
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
        axes[1, 1].plot(valeurs, iterations, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('max_iter')
        axes[1, 1].set_ylabel('Itérations réelles')
        axes[1, 1].set_title('Itérations réelles vs max_iter')
    else:
        # Calcul du score combiné pour chaque valeur
        scores_combines = []
        for i in range(len(valeurs)):
            score = (0.6 * silhouettes[i]) + (0.3 * (calinskis[i] / 1000)) + (0.1 * (1 / davies[i]))
            scores_combines.append(score)
        
        axes[1, 1].plot(valeurs, scores_combines, 'co-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel(parametre)
        axes[1, 1].set_ylabel('Score Combiné')
        axes[1, 1].set_title('Score Combiné vs ' + parametre)
    
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(meilleure_valeur, color='red', linestyle='--', alpha=0.7, label=f'Meilleur: {meilleure_valeur}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"🎯 MEILLEUR {parametre}: {meilleure_valeur}")
    return meilleure_valeur

def visualiser_clusters_finaux(points, config_optimale):
    """Visualise les clusters finaux avec la configuration optimale"""
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
    
    # Graphique des métriques
    plt.subplot(1, 2, 2)
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
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def tester_hyperparametres_complet(chemin_fichier):
    """Test complet avec visualisations graphiques"""
    
    # Charger les points
    points = lire_fichier_arff(chemin_fichier)
    print(f"✅ {len(points)} points chargés")
    print("📊 n_init=1 pour voir les vraies différences entre les méthodes")
    
    # Configuration de base avec n_init=1
    config = {'n_init': 1, 'random_state': None}
    
    # 1. Test n_clusters
    meilleur_k = tester_hyperparametre(
        points, 
        'n_clusters', 
        [2, 3, 4, 5, 6],
        config
    )
    config['n_clusters'] = meilleur_k
    
    # 2. Test init
    meilleure_init = tester_hyperparametre(
        points,
        'init',
        ['k-means++', 'random'],
        config
    )
    config['init'] = meilleure_init
    
    # 3. Test max_iter
    meilleur_max_iter = tester_hyperparametre(
        points,
        'max_iter',
        [10, 50, 100, 200, 300, 400, 500],
        config
    )
    config['max_iter'] = meilleur_max_iter
    
    # Résumé final et visualisation
    print("\n" + "=" * 50)
    print("🎯 CONFIGURATION OPTIMALE")
    print("=" * 50)
    for param, valeur in config.items():
        print(f"{param}: {valeur}")
    
    # Visualisation finale des clusters
    print("\n📈 VISUALISATION DES CLUSTERS FINAUX")
    visualiser_clusters_finaux(points, config)

# Test dans le main
if __name__ == "__main__":
    tester_hyperparametres_complet("/home/boaglio/5A/clustering/Clustering/src/dataset/artificial/2d-20c-no0.arff")