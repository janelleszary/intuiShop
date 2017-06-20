import re
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD



names = ['id','category']
for i in range(1,301):
	names.append(str(i))
	
	

# Load text description vectors
descriptions = pd.read_csv('data/description_vecs.csv',names=names)
X = descriptions[descriptions.columns[2:302]]

# Reduce dimensionality with tSVD?
# X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
# No, dimensionality isn't the problem... it's sample size
# Reduce sample size:

X_reduced=pd.DataFrame()
for i in range(0,len(X)):
	if (i%100==0):
		X_reduced = X_reduced.append(X[:i])



# T-SNE dimensionality reduction
X_embedded = TSNE(n_components=2, verbose=2, metric='cosine').fit_transform(X_reduced)
# plt.scatter(X_embedded[:,0], X_embedded[:,1], 100, c=1)

