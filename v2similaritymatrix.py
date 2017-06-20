import re
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


names = ['id','category']
for i in range(1,301):
	names.append(str(i))
	
	
# Load text description vectors
descriptions = pd.read_csv('data/description_vecs.csv',names=names)
X = descriptions[descriptions.columns[2:302]]


# Decrease sample size. Keep 1/100th
X_reduced=pd.DataFrame()
for i in range(0,len(X)):
	if (i%100==0):
		X_reduced = pd.concat([X_reduced, X[i:i+1]])

X = X_reduced
	
with open('data/similarity_matrix_100.csv', 'w') as f:
	for i in range(0,len(X)):
		myrow = []
		for j in range(0,len(X)):
			mycos = cosine_similarity(X[i:i+1], X[j:j+1])[0][0]
			myrow.append(mycos)
		myrow = str(myrow)[1:-1]
		a = f.write(myrow)
		a = f.write('\n')		
		if (i%100==0):
			print(i)


f.close()


