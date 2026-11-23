# Recommandation de films (MovieLens 100K)
## Données

**Source :** MovieLens 100K, GroupLens Research (Université du Minnesota).
Page : https://grouplens.org/datasets/movielens/100k/
Les données ne sont pas versionnées : télécharger `ml-100k.zip` et extraire
`u.data`, `u.item` et `README` dans `data/ml-100k/`.

**Conditions d'utilisation (extraites du README de MovieLens) :**
- usage pour la recherche uniquement, pas d'usage commercial sans permission ;
- redistribution interdite sans autorisation (d'où le `.gitignore` sur `data/`) ;
- ne pas laisser entendre un soutien de l'Université du Minnesota ou de GroupLens.

**Citation :** F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens
Datasets: History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19. DOI=http://dx.doi.org/10.1145/2827872

**Colonnes utilisées :**
- `u.data` (séparateur tabulation) : `user_id`, `item_id`, `rating` (1 à 5), `timestamp`
- `u.item` (séparateur `|`, encodage latin-1) : `item_id`, `title`

**Contrôles qualité :** aucune valeur manquante, aucun doublon, notes entre 1 et 5,
tous les films notés présents dans le catalogue. Aucun nettoyage n'a été nécessaire.
18 titres sont en double dans `u.item` : l'`item_id` sert donc d'identifiant.

**Profils éligibles :** utilisateur ayant au moins 2 films aimés (note ≥ 4),
soit 942 profils sur 943.