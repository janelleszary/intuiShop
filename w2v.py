# Collect vectors for each word in description, average into description-level vectors
# You need to have word2vec pretrained vectors downloaded, and nltk stopwords 

from nltk.corpus import stopwords
import nltk
import gensim
import numpy as np
import pandas as pd
import re


# Load pretrained w2v from GoogleNews vectors
print('Importing model...')
fn = '../../Academic_Work/PROJECTS/corpora/animal/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(fn, binary=True)


print('Importing data...')
mydata = pd.read_csv('data/data1kthreshold.csv')

print('Creating vectors...')
with open('data/description_vecs.csv', 'w') as vec_file:
	for i in range(0,len(mydata)):
		mymat = []
		myd = mydata.description[i]
		myd = myd.lower().split()
		words = [w for w in myd if not w in stopwords.words("english")]
		for word in words:
			if word in model.vocab:
				mymat.append(model[word])
		myvec = np.mean(mymat, axis=0)
		try:
			if (len(myvec)==300):
				vecstring = []
				vecstring = [str(vec) for vec in myvec]
				newline = (mydata.id[i] + ', ' + mydata.category1[i] + ', ' + str(vecstring))
				newline = re.sub('\[', '', newline)
				newline = re.sub('\]', '', newline)
				newline = re.sub('\'', '', newline)
				vec_file.write('%s\n' % (newline))
		except:
			pass

print('Done.')
