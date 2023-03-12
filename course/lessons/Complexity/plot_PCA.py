"""
Categorising Friends
====================

In this section we look at an example of dimension reduction 
using a method known as principle component analysis.

"""


##############################################################################
# David's friends
# ---------------
#
# Here we study David's friends posting on Facebook. In 2017, David
# looked at the posts made by each of his friends and categorised then
# according to what they posted about: Work, Adverts, Culture etc.
# 
# Let's start by printing out a matrix showing the proportion of posts by
# 32 of his friends (labelled A, B, C, etc.) on the different categories

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import display

# Load friends data. 
friends=pd.read_csv('FacebookFriends.csv',';',index_col=[0])

# Adjust to proportions. 
friends = friends.apply(lambda x: x/ x.sum(), axis=0).transpose()
# Print out frineds (rows) and post categories (columns)
display(friends)


##############################################################################
# Principal components
# --------------------
#
# Principle component analysis is a dimension reduction method.
# 


num_components=2

pca = PCA(n_components=num_components)
X = pca.fit(friends).transform(friends)

for component in range(num_components):
    fig, ax = plt.subplots()
    objects = friends.columns
    y_pos = np.arange(len(objects))+1
    ax.barh(y_pos, pca.components_[component,:], align='center', alpha=0.5)
    #ax.set_facecolor((1,1,1))
    ax.set_yticks(np.arange(-1,50,step=50))
    ax.set_ylim(0,len(objects)+1) 
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # placing each of the x-axis labels individually
    for i, y_position in enumerate(pca.components_[component,:]):
        if y_position < 0:
            ax.text(y_position-0.01,i+1.35, objects[i], ha="right", va="top")
        else:
            ax.text(y_position+0.01,i+1.35, objects[i], ha="left", va="top")
    ax.set_xlabel('Contribution to component %d' % (component+1))

    plt.show()


##############################################################################
# Displaying the friends
# ----------------------
#
# Plot the friends in two dimensions.
# 


plt.figure()
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.8)
ax.legend(loc="best", shadow=False, scatterpoints=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel('Component 1 (Public vs. Personal)' )
ax.set_ylabel('Component 2 (Other interests vs. Work)' )
plt.show()



##############################################################################
# Categorising the friends
# ------------------------
#
# 
# 

args = {
    "init": "random",
    "n_init": 10,
    "max_iter": 20,
    "random_state": 30,
}

kmeans = KMeans(n_clusters=3, **args).fit(X)
classes=kmeans.predict(X)
keys=dict({0: 'x' ,1: 'o',2: 's'})


plt.figure()
fig, ax = plt.subplots()
for i in np.unique(classes):
    Xi = X[classes==i,:]
    ax.scatter(Xi[:, 0], Xi[:, 1], marker=keys[i], alpha=0.8)
    ax.legend(loc="best", shadow=False, scatterpoints=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel('Component 1 (Public vs. Personal)' )
    ax.set_ylabel('Component 2 (Other interests vs. Work)' )
 
plt.show()


