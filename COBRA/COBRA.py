# Creating a class COBRA , which take sklearn estimators as input and return a
# COBRA object

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from .helper import *


class COBRA(BaseEstimator):
    """
    A simple COBRA implementation of COBRA Techniques, which took input of a
    sklearn estimator and return a COBRA object.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator to be used in the COBRA.
    """

    def __init__(self, estimators,debug = False):
        self.estimators = estimators
        self.debug = debug
        self.estimator_names = [estimator.__class__.__name__ for estimator in self.estimators]
        self.n_estimator = len(self.estimators)

    def fit(self, X, y ,split = 0.5,random_state = 0):
        """
        This function first of all split the data into D_k and D_l, where D_k
        is the training data and D_l is the test data. Then, it fit the
        estimator on D_k and predict the labels of D_l.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        split : float, optional (default = 0.5)
            The ratio of the training data and the test data.
        random_state : int, optional (default = 0)
            The random seed.
        """

        Dk, Dl, Dyk, Dyl = train_test_split(X, y,test_size=split,
                                                    random_state=random_state)

        #An array of estimators, fitted on the Dk
        fitters = [estimator.fit(Dk,Dyk)  for estimator in self.estimators]
        self.fitters = fitters
        #Prediction on the set Dl
        prediction = np.array([fitter.predict(Dl) for fitter in fitters]).T
        self.prediction_on_Dl = prediction

        #DEBUGGING
        if self.debug :
            print(f"Shape of prediction : {prediction.shape}")
            print(f"Length of Estimators : {len(self.estimators)}")
            print(f"Shape of Dl : {Dl.shape}")
            print(f"Shape of Dyk : {Dyk.shape}")
            print(f"Shape of Dyl : {Dyl.shape}")
            print(f"Shape of Dk : {Dk.shape}")
            print(f"Shape of fitters : {len(self.fitters)}")


        return self

    def predict(self, X , epsilon = 0.1 , alpha = 0.5):
        """
        This function takes the query data and return the predicted labels.

        ----Attributes----
        epsilon : float, optional (default = 0.1)
            The epsilon parameter for the COBRA.
        alpha : float, optional (default = 0.5)
            The alpha parameter for the COBRA.
        returns : array-like, shape = [n_samples]
        """

        prediction_on_query = np.array([fitter.predict(X) for fitter in
                                  self.fitters]).T
        prediction_on_Dl = self.prediction_on_Dl
        if self.debug : print(f"Shape of prediction_on_query : {prediction_on_query.shape}")

        distance = dist(prediction_on_query,prediction_on_Dl,self.debug)
        Weight = weight(distance,epsilon)
        Weight = Weight < alpha * self.n_estimator
        pred = np.array([np.mean(Dyl[w]) for w in Weight])
        return pred