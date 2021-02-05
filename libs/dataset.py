from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

random_state = 42
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def create_simple_dataset(num_sample = 100,separation=1):
    X, y = make_classification(n_samples=num_sample,
                                n_features=2, 
                                n_redundant=0, 
                                n_informative=2,
                                random_state=random_state, 
                                n_clusters_per_class=1,
                                class_sep=separation)
    
    return X,y


def create_moon_dataset(num_sample = 100,separation=0.3):
    X, y = make_moons(noise=separation, 
                        n_samples =num_sample,
                        random_state=random_state)
    
    return X,y


def create_circle_dataset(num_sample = 100,separation=0.3,factor = 0.5):
    X, y = make_circles(noise=separation, 
                            factor=factor, 
                            random_state=random_state,
                            n_samples=num_sample)

    return X,y



def plot_dataset(X,y):
    fig1 = plt.figure(figsize=(5,4),dpi=150)
    ax1 = fig1.add_axes([0.1,0.1,0.9,0.9])
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')

def plot_decision_region(X,y,classifier,resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','green','gray','cyan')
    
    classes = np.unique(y)
    cmap = ListedColormap(colors[:len(classes)])

    x1_min,x1_max = X[:,0].min() - 1,X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() - 1,X[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    for idx,cl in enumerate(classes):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl)

    plt.legend(loc="upper left")


def plot_decision_region_proba(X,y,classifier,resolution=0.02):

    cm = plt.cm.RdBu
    cm_bright = ['#FF0000', '#0000FF']

    classes = np.unique(y)
    markers = ('s','x')

    x1_min,x1_max = X[:,0].min() - 1,X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() - 1,X[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z = classifier.predict_proba(np.array([xx1.ravel(), xx2.ravel()]).T)[:, 1]
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cm)

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    for idx,cl in enumerate(classes):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cm_bright[idx],marker=markers[idx],label=cl)

    plt.legend(loc="upper left")



