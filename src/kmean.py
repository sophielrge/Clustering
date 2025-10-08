import numpy as np
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
    Teste un hyperparamètre avec différentes valeurs
    """
    if config_base is None:
        config_base = {'n_clusters': 3, 'init': 'k-means++', 'max_iter': 300, 'n_init': 1}  # n_init=1 !
    
    print(f"\n🧪 TEST {parametre}")
    print("=" * 50)
    
    # En-tête du tableau
    if parametre == 'n_clusters':
        print("K   | Silhouette | Calinski | Davies")
    elif parametre == 'init':
        print("Méthode   | Silhouette | Calinski | Davies")
    elif parametre == 'max_iter':
        print("max_iter | Silhouette | Calinski | Davies | Itérations")
    else:
        print(f"{parametre:8} | Silhouette | Calinski | Davies")
    
    print("-" * 50)
    
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
        davies = davies_bouldin_score(points, labels)
        
        # Afficher les résultats
        if parametre == 'n_clusters':
            print(f"{valeur:2} | {silhouette:9.3f} | {calinski:7.1f} | {davies:6.3f}")
        elif parametre == 'init':
            print(f"{valeur:9} | {silhouette:9.3f} | {calinski:7.1f} | {davies:6.3f}")
        elif parametre == 'max_iter':
            iterations = kmeans.n_iter_
            print(f"{valeur:8} | {silhouette:9.3f} | {calinski:7.1f} | {davies:6.3f} | {iterations:10}")
        else:
            print(f"{valeur:8} | {silhouette:9.3f} | {calinski:7.1f} | {davies:6.3f}")
        
        # Score combiné
        score_combiné = (0.6 * silhouette) + (0.3 * (calinski / 1000)) + (0.1 * (1 / davies))
        
        if score_combiné > meilleur_score:
            meilleur_score = score_combiné
            meilleure_valeur = valeur
    
    print(f"🎯 MEILLEUR {parametre}: {meilleure_valeur}")
    return meilleure_valeur

def tester_hyperparametres_complet(chemin_fichier):
    """Test complet avec n_init=1 pour voir les vraies différences"""
    
    # Charger les points
    points = lire_fichier_arff(chemin_fichier)
    print(f"✅ {len(points)} points chargés")
    print("📊 n_init=1 pour voir les vraies différences entre les méthodes")
    
    # Configuration de base avec n_init=1
    config = {'n_init': 1, 'random_state': None}  # Pas de random_state fixe pour voir la variabilité
    
    # 1. Test n_clusters
    meilleur_k = tester_hyperparametre(
        points, 
        'n_clusters', 
        [2, 3, 4, 5, 6],
        config
    )
    config['n_clusters'] = meilleur_k
    
    # 2. Test init - MAINTENANT on devrait voir des différences !
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
        [10, 50, 100, 200, 300],
        config
    )
    config['max_iter'] = meilleur_max_iter
    
    # Résumé final
    print("\n" + "=" * 50)
    print("🎯 CONFIGURATION OPTIMALE")
    print("=" * 50)
    for param, valeur in config.items():
        print(f"{param}: {valeur}")

# Test dans le main
if __name__ == "__main__":
    tester_hyperparametres_complet("/home/boaglio/5A/clustering/Clustering/src/dataset/artificial/2d-10c.arff")