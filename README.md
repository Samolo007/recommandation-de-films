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

## Modèle

**Problème :** à partir d'un film aimé, proposer 5 films proches.

**Découpage train/test** (`src/split.py`) :
- film « aimé » = note ≥ 4 ; profil éligible = au moins 2 films aimés (942 profils sur 943) ;
- film masqué = dernier film aimé du profil (tri par `timestamp`, puis par `item_id`
  en cas d'égalité, ce qui concerne 305 profils) ;
- film de départ = avant-dernier film aimé ;
- le film masqué est retiré de `train` : 99 058 notes en entraînement, 942 profils de test.

**Référence simple** (`src/baseline.py`) : les 5 films les plus aimés dans `train` (le
plus de notes ≥ 4), sans le film de départ.

**Modèle de base** (`src/knn.py`), le baseline demandé par le prof : k-NN par similarité
cosinus. Chaque film est décrit par l'ensemble des utilisateurs qui l'ont aimé ; deux
films sont proches si les mêmes utilisateurs les ont aimés. L'index est calculé sur
`train` uniquement.

**Modèle A** (`src/knn.py`, fonction `build_neighbors_svd`) : on part du même tableau
films × spectateurs (notes ≥ 4), mais chaque film est d'abord résumé en 20 nombres
trouvés automatiquement (SVD tronquée, `random_state=0`). Les 5 voisins sont ensuite
cherchés avec le cosinus sur ces 20 nombres. Le nombre 20 a été choisi parmi 10, 20, 50,
100 et 200 sur une répétition générale tirée de `train` : le test n'a pas servi à ce
choix. Mêmes données, même format d'index.

**Métrique :** Hit@5 = part des profils dont le film masqué est parmi les 5 films
proposés à partir du film de départ. Même test, même métrique pour les trois modèles.

## Résultats

| Modèle | Profils trouvés | Hit@5 |
|---|---|---|
| Top 5 populaire | 16 sur 942 | 0,0170 |
| Modèle de base (k-NN cosinus) | 44 sur 942 | 0,0467 |
| Modèle A (SVD, 20 curseurs) | 49 sur 942 | 0,0520 |

Exemples (avec le modèle de base) : *Aliens (1986)* → *Alien (1979)* trouvé ;
*M\*A\*S\*H (1970)* → *Butch Cassidy and the Sundance Kid (1969)* trouvé. Raté : depuis
*Schindler's List (1993)*, le film masqué était *Paradise Lost: The Child Murders at Robin Hood Hills (1996)*.

Les deux modèles de voisins font nettement mieux que le top 5 populaire. Le modèle A
retrouve 5 films cachés de plus que le modèle de base, mais l'écart est petit : 21
profils sont trouvés seulement par le modèle A, 16 seulement par le modèle de base,
28 par les deux et 877 par aucun (test de McNemar : p ≈ 0,51, un écart compatible avec
le hasard). À la répétition générale, l'écart était plus net (65 contre 47) ; il ne
s'est pas retrouvé à cette ampleur sur le vrai test. Résultat publié tel quel : aucun
réglage n'a été modifié après l'avoir vu.

Pour *Toy Story (1995)*, le modèle de base propose surtout de gros succès (*Star Wars*,
*Return of the Jedi*, *Raiders of the Lost Ark*…), le modèle A des films d'animation
familiale (*Hercules*, *Aladdin*, *Beauty and the Beast*…).

## Limites

- Le gain du modèle A sur le modèle de base n'est pas établi sur le vrai test (p ≈ 0,51).
- Les 20 nombres du modèle A ne se lisent pas directement.
- 877 profils sur 942 sont ratés par les deux modèles : retrouver un film précis parmi
  1 682 avec 5 propositions reste difficile.
- Un seul film masqué et un seul film de départ par profil : le Hit@5 est une
  estimation sur 942 cas.
- Pour 305 profils, plusieurs films aimés ont le même `timestamp` : le « dernier »
  film est un choix reproductible mais arbitraire.
- Le modèle de base retrouve mieux les films très aimés : 0,091 quand le film masqué est
  dans le top 100 (209 profils, 19 trouvés), 0,034 sinon (733 profils, 25 trouvés).
- 252 films n'ont aucune note ≥ 4 dans `train` : ils sont dans le catalogue mais sans
  voisins.
- 18 titres sont en double dans le catalogue : par exemple *Fly Away Home (1996)*
  existe sous deux `item_id` (304 et 500), et chacun est le premier voisin de l'autre.

## Export et reproduction

`python src/train_model.py` génère `models/knn_index.joblib` à partir du **modèle A**
(SVD à 20 curseurs, `random_state=0` : le même fichier à chaque exécution). Le fichier
n'est pas versionné (voir la licence MovieLens) : `{"neighbors": ..., "catalog": ...}`.
Python 3.14.

    python -m pip install -r requirements.txt
    python src/evaluate.py
    python src/train_model.py


## API
### Lancer l'API

Toutes les commandes se lancent depuis recommandation-de-films.

Prérequis : les données dans `data/ml-100k/` (voir la section Données) et les dépendances installées :
```
 python -m pip install -r requirements.txt
```

1. Générer l'index des voisins (une seule fois) :
   ```
   python src/train_model.py
   ```
   Cette commande crée `models/knn_index.joblib`. Sans ce fichier, l'API refuse de démarrer.
2. Lancer l'API :
   ```
   python -m uvicorn api:app --reload
   ```
3. Tester l'API dans le navigateur : `http://127.0.0.1:8000/docs`

### Lancer les tests

    python -m pytest

Attendu : `10 passed`. Les tests utilisent un faux index de 7 films : ils n'ont besoin ni des données ni de `models/`.

La CI GitHub Actions (`.github/workflows/tests.yml`) lance ces mêmes tests à chaque push et à chaque Pull Request, avec Python 3.14.

### GET /health

Vérifier si l'API fonctionne

```json
{"status": "ok"}
```

### POST /predict

#### Entrée

```json
{"item_id": 50}
```

- `item_id` : entier strictement positif (identifiant MovieLens du film).
- Refusés avec 422 : texte (`"50"`), nombre à virgule (`50.5`), valeur ≤ 0, champ manquant ou en trop.

#### Réponse

Film avec voisins (200) :

```json
{
  "item_id": 50,
  "title": "Star Wars (1977)",
  "recommendations": [
    {"item_id": 181, "title": "Return of the Jedi (1983)", "score": 0.970},
    {"item_id": 172, "title": "Empire Strikes Back, The (1980)", "score": 0.890},
    {"item_id": 174, "title": "Raiders of the Lost Ark (1981)", "score": 0.854},
    {"item_id": 1, "title": "Toy Story (1995)", "score": 0.811},
    {"item_id": 210, "title": "Indiana Jones and the Last Crusade (1989)", "score": 0.807}
  ],
  "message": null
}
```

Les 5 films sont triés du plus proche au moins proche, et le film choisi n'en fait jamais partie. `score` = similarité cosinus, entre 0 et 1 (arrondie dans cet exemple).

Les scores dépendent du modèle : ceux du modèle A (cosinus sur les 20 nombres) ne se
comparent pas à ceux du modèle de base.

Film sans voisins (200) :

```json
{
  "item_id": 1407,
  "title": "Specialist, The (1994)",
  "recommendations": [],
  "message": "Pas de recommandation : aucun spectateur n'a donné 4 ou 5 à ce film"
}
```

Code 200 : la demande est valide, car le film existe dans le catalogue. Il n'a simplement aucun
voisin : personne ne lui a donné 4 ou 5 dans les données d'entraînement (252 films sur 1 682).

#### Erreurs

| Cas | Code | Réponse |
|---|---|---|
| `item_id` absent du catalogue | 404 | `{"detail": "Film introuvable : cet identifiant n'existe pas dans le catalogue."}` |
| Entrée invalide | 422 | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |

## Exemple de test

- `{"item_id": 99999}` → 404 , `{"detail": "Film introuvable : cet identifiant n'existe pas dans le catalogue."}`
- `{"item_id": 0}` → `422`, avec `loc` = `["body", "item_id"]`