# Optimiseur
```
Algorithme qui permet de minimiser la fonction de perte pondérée.

SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
Descente de gradient stochastique (SGD)(modifié)
Momentum : Momentum prend en compte les gradients passés pour aplanir les étapes de la descente. Il peut être appliqué avec une descente par gradient en batch, une descente en gradient par mini-batch ou une descente par gradient stochastique.
batch_size indique la taille du sous-ensemble de votre échantillon d'apprentissage (par exemple, 100 sur 1 000) qui sera utilisée pour former le réseau au cours de son processus d'apprentissage. Chaque lot entraîne le réseau dans un ordre successif, en tenant compte des poids mis à jour provenant de l'appliance du lot précédent. return_sequence indique si une couche récurrente du réseau doit renvoyer la totalité de sa séquence de sortie (c'est-à-dire une séquence de vecteurs de dimension spécifique) à la couche suivante du réseau, ou tout simplement sa dernière sortie, qui est un vecteur unique de la même dimension. Cette valeur peut être utile pour les réseaux conformes à une architecture RNN. batch_input_shape définit que la classification séquentielle du réseau de neurones peut accepter des données d'entrée de la taille de lot définie uniquement, limitant ainsi la création de tout vecteur de dimension variable. Il est largement utilisé dans les réseaux LSTM empilés.
```

# Dropout
```
Dropout prend la sortie des activations de la couche précédente et définit de manière aléatoire une certaine fraction (taux d'abandon) des activations sur 0, en les annulant ou en les «supprimant».

C'est une technique de régularisation courante utilisée pour prévenir les surajustements dans les réseaux de neurones. Le taux d'abandon est l'hyperparamètre ajustable qui est ajusté pour mesurer les performances avec différentes valeurs. Il est généralement défini entre 0,2 et 0,5 (mais peut être défini de manière arbitraire).
Le décrochage n’est utilisé que pendant la formation ; Au moment du test, aucune activation n'est abandonnée, mais réduite par un facteur de taux d'abandon. Cela tient compte du nombre d'unités actives pendant la période de test par rapport à la durée de formation.
```

```
Lr : Le taux d'apprentissage est un hyper-paramètre qui contrôle à quel point nous ajustons les poids de notre réseau en fonction du gradient de perte. Plus la valeur est basse, plus nous roulons lentement sur la pente descendante. Bien que cela puisse être une bonne idée (en utilisant un taux d’apprentissage faible) pour nous assurer que nous ne manquons aucun minimum local, cela pourrait également signifier que nous allons prendre beaucoup de temps pour converger - surtout si nous restons coincés. une région de plateau.

new_weight = existing_weight — learning_rate * gradient
```

# Fonctions de perte dans les réseaux de neurones


## The currently available loss functions for Keras are as follows:
```
mean_squared_error
mean_absolute_error
mean_absolute_percentage_error
mean_squared_logarithmic_error
squared_hinge
hinge
categorical_hinge
logcosh
categorical_crossentropy
sparse_categorical_crossentropy
binary_crossentropy
kullback_leibler_divergence
poisson
cosine_proximity
```


![epoch](https://github.com/Harisonm/4aibd-s1-project-ml/blob/master/docs/pictures/epoch.png)







