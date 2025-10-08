1. Coefficient de Silhouette

a(i) = distance moyenne aux autres points du même cluster

    b(i) = distance moyenne aux points du plus proche cluster voisin

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

Interprétation

    +1 : Excellent clustering (points bien regroupés)

    0 : Clusters qui se chevauchent

    -1 : Mauvais clustering (points mal assignés)



2. Indice de Calinski-Harabasz
   Concept

Mesure le rapport entre la dispersion inter-clusters et intra-cluster.
Calcul
text

CH = [B / (k-1)] / [W / (n-k)]

Où :

    B = somme des dispersions inter-clusters

    W = somme des dispersions intra-clusters

    k = nombre de clusters

    n = nombre total de points



. Indice de Davies-Bouldin
Concept

Mesure la similarité moyenne entre chaque cluster et son cluster le plus similaire.
Calcul

Pour chaque cluster i :

    R(i) = max[(s(i) + s(j)) / d(c(i), c(j))] pour j ≠ i
    Où :

    s(i) = dispersion moyenne dans le cluster i

    d(c(i), c(j)) = distance entre centroïdes

DB = (1/k) * Σ R(i)
Interprétation

    Valeur faible = meilleur clustering

    0 = clustering parfait