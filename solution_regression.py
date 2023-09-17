# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

import numpy as np
import random
import gestion_donnees
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        phi_x = np.array([])
        # AJOUTER CODE ICI
        if isinstance(x, (float,int)):
            # Si x est un scalaire
            phi_x = np.array([x**i for i in range(1, self.M + 1)])
        elif (isinstance(x, np.ndarray) and x.ndim==1 ):
            # Si x est une liste de scalaires
            phi_x = np.array([[y**i for i in range(1, self.M + 1)] for y in x])
        # phi_x = x
        return phi_x

    def recherche_hyperparametre(self, X, t,k=10):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée 
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k", 
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties."""
        

        indices_shuffled = list(range(len(t)))
        random.shuffle(indices_shuffled)
        X = X[indices_shuffled]
        t = t[indices_shuffled]

        if k>len(t) :
            k = len(t)

        X_folds = np.array_split(X,k)
        t_folds = np.array_split(t,k)

        m_values = [1, 2, 3, 4, 5]
        avg_scores = []
        best_m = None
        best_score = float('inf')

        for m in m_values :

            model = linear_model.LinearRegression()
            scores = []
            for i in range(k):

                X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
                X_valid = X_folds[i]

                t_train = np.concatenate([t_folds[j] for j in range(k) if j != i])
                t_valid = t_folds[i]

                X_train_poly = np.power(X_train, m)
                X_test_poly = np.power(X_valid, m)

                X_train_poly = X_train_poly.reshape(-1, 1)

                model.fit(X_train_poly, t_valid)

                t_pred = model.predict(X_test_poly)

                metric = np.mean((t_valid - t_pred) ** 2)

                scores.append(metric)
            
            avg_scores.append(np.mean(scores))

            if np.mean(scores) < best_score:
                best_m = m
                best_score = np.mean(scores)
        
  
        
        return best_m









        """
        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note: 

        Le resultat est mis dans la variable self.M
        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI
        self.M = 1

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)
        self.w = [0, 1]

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI
        return 0.5

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI
        return 0.0


    

test = Regression(5,5)
data = gestion_donnees.GestionDonnees([1,2,3,4,5],"linear",50,10,2)

x_train, t_train, x_test, t_test = data.generer_donnees()
result = test.recherche_hyperparametre(x_train,t_train)
print(result)
