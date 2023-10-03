# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

import numpy as np
import random
from sklearn import linear_model
from sklearn.utils import shuffle



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
        # AJOUTER CODE ICI
        
        '''phi_x = []

        if isinstance(x, (float,int)):
            # Si x est un scalaire
            phi_x = [x**i for i in range(1, self.M + 1)]
        elif isinstance(x, np.ndarray):
            # Si x est une liste de scalaires
            phi_x = np.array([[y**i for i in range(1, self.M + 1)] for y in x])
        return (phi_x)'''
        if(type(x) == np.ndarray):  #Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM
            phi_x = []
            for i in x:
                phi_x += [[i**j for j in range(1, self.M + 1)]]
            phi_x = np.array(phi_x)
        else:                       #Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
            phi_x = np.array([x**i for i in range(1, self.M + 1)])

        return phi_x
    
        '''if x.ndim == 0:
            phi_x = [x**i for i in range(1, self.M + 1)]
            # phi_x = x ** np.arange(1, self.M+1 )
        else:
            # phi_x = np.empty([np.shape(x)[0], self.M])
            phi_x = np.ones([np.shape(x)[0], self.M])
            phi_x = [[y**i for i in range(1, self.M + 1)] for y in x]
            phi_x=np.array(phi_x)'''

        return phi_x
    

    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée 
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k", 
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties."""

        k=10

        '''data = np.concatenate((X,t.T))
        random.shuffle(data)'''

        if k > len(X): 
            k = len(X)
        
        X_shuffled, t_shufled = shuffle(X, t, random_state=0)
        X_folds = np.array_split(X_shuffled, k)
        t_folds = np.array_split(t_shufled, k)
        
        '''data_folds = np.array_split(data,k)
        data_folds = np.array(data_folds)
        print(data_folds.shape)'''
        average_errors = {}
        m_values = [i for i in range(1, 11)]
        min_value = float('inf')
        self.M = 1

        for m in m_values : 

            all_mse = []
            for i in range(k): 
                
                X_train = np.delete(X_folds,i)
                X_valid = X_folds[i]

                t_train = np.delete(t_folds,i)
                t_valid = t_folds[i]
              

                self.entrainement(X_train, t_train, False)

                predictions = np.array([self.prediction(x_valid) for x_valid in X_valid])

                MSE = np.mean(self.erreur(t_valid, predictions))
                all_mse.append(MSE)

            average_errors[m] = np.mean(all_mse)

        for key, value in average_errors.items():
            if value < min_value: 
                min_value  = value
                self.M = key
        
        return self.M

    
    
        """
        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note: 

        Le resultat est mis dans la variable self.M
        X: vecteur de donnees
        t: vecteur de cibles """
        

    def entrainement(self, X, t, using_sklearn=False):
        
        """Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
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
        

        if using_sklearn == True:
            reg = linear_model.Ridge(alpha = self.lamb)
            reg.fit(phi_x, t)
            self.w = reg.coef_
            self.w[0] = reg.intercept_
        
        else:
            I = np.identity(phi_x.shape[1])
            produit_phi = np.dot(np.transpose(phi_x),phi_x)
            t = np.array(t)
            self.w = np.linalg.solve(self.lamb * I + produit_phi, np.transpose(phi_x).dot(t) )

        #return self.w

        

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI

        phi_x = self.fonction_base_polynomiale(x)
        return np.dot(np.transpose(self.w), phi_x)

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI
        return (prediction - t) ** 2



