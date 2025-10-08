### Elian BOAGLIO

### Sophie LARGE

### 5 SDBD B1

# Clustering

## K-means

### Principe de la méthode

La méthode k-means est une méthode de partionnement qui a pour objectif de minimiser la distance intra-cluster.
Cette méthode permet de partager un ensemble de points en un nombre donnée de clusters (k). Chaque cluster sera par la suite représenter par un unique point. Cette méthode s'appuie sur l'algothime de Lloyd. \n

Il existe plusieurs déclinaisons de cette méthode :
- **k-means random** : Ses centres ne sont pas forcément des points appartenant au jeu de données.
- **k-means++** : On initialise de manière plus intelligente, c'est-à-dire que les centres initiaux sont choisis ded façon optimale.
- **mini-batch** : Cette méthode est destinée aux jeux de données avec un grand nombre de points. Elle séquence les points en plus petits lots pour ensuite calculer les centres de manière plus optimale.
- **k-medoids** : Elle repose sur l'algorithme "Permutation Around Medoids". Son objectif est de minimiser la somme des erreurs absolues aux k medoïds. Les centres finaux des clusters correspondent à des points du jeu de données. 

### Étape de la méthode
1. **Initialisation** : Positionne les centres initiaux des k clusters, soit avec la méthode "k-means++" soit avec celle "random".
2. **Création de groupes** : Pour chaque point du jeu de données, l'algorithme l'affecte à un des clusters créés précédemment. 
3. **Mise à jour** : On recalcule les centres de chaque cluster.
4. **Attente qu'une condition d'arrêt soit complétée** :  


### Hyperparamètres
- **n_clusters ou k** : \n
C'est le nombre de cluster que l'on veut.
- **init** : \n
Ce paramètre permet de choisir la méthode d'initalisation des centroïdes. \n
Ses valeurs possibles sont :
    - _random_ : choisit aléatoirement les centres initiaux. Cette méthode est plus rapide mais moins fiable.
    - _k-means++ (par défaut)_ : calcule en amont les écarts entre chaque centre inital pour les placer de manière optimale.
- **n_init** : \n 
Cela correspond au nombre de graines aléatoires pour l'initialisation.
- **max_iter** : \n
Avec ce paramètre, nous pouvons limiter le nombre d'itérations de notre recherche. Il permet d'éviter les boucles infinis. 
- **tol** : \n
C'est une condition d'arrêt qui repose sur la stabilisation des centres, c'est-à-dire que l'algorithme s'arrête lorsque les centres ne bougent presque plus. 
- **algorithm** :  \n 
Ici nous choississons l'algorithme qui réalisera le clustering. Par défaut, l'algorithme choisi est celui de Lloyd.

### Hyperparamètres testés

