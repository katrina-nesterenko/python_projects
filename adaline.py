import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
    """
    ADAptive LInear NEuron classifier
    
    parameters
    ----------
    eta : float
        learning rate (between 0.0 and 1.0)
    n_iter : int
        number of passes over the training dataset
    
    attributes
    ----------
    w_ : 1d-array
        weights after fitting
    errors_ : list
        number of misclassifications in every epoch
    """
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y): 
        """
        Fit training data

        parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            training vectors, where n_samples is the number of samples and
            n_features is the number of features
        y : array-like, shape = [n_samples]
            target values
        
        returns
        -------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) # weights 1 to m
            self.w_[0] += self.eta * errors.sum() # the zero-weight
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        Compute linear activation
        """
        return self.net_input(X)

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)

# read iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None) # dataframe object
#print(df.tail())
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

"""
# plot different learning rates
fig, ax = plt.subplots(nrows = 1, ncols=2, figsize = (8, 4))
ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('adaline - learning rate 0.01')

ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('sum-squared-error')
ax[1].set_title('adaline - learning rate 0.0001')
plt.show()
"""
# plot decision regions (same function from perceptron)
def plot_decision_regions(X, y, classifier, resolution = 0.2):
    # set up marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('plum', 'mediumturquoise', 'mediumseagreen', 'lavenderblush', 'mintcream')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)

# feature scaling - standardization
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# retrain using learning rate n = 0.01
ada = AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier = ada)
plt.title('adaline - gradient descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('epochs')
plt.ylabel('sum-squared-error')
plt.show()




# --- plot raw data ---
"""
plt.scatter(X[:50, 0], X[:50, 1], color = 'plum', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'skyblue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show()
"""

# --- plot error convergence ---
"""
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('epochs')
plt.ylabel('number of misclassifications')
plt.show()
"""

"""
# contour plot of decision regions
plot_decision_regions(X, y, classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()
"""