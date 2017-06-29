'''
Uses word embeddings from pretrained gensim W2V model, trained on GoogleNews.
Sums embeddings for each word in item descriptions to output a w2v embedding for each description.

Requires:
- pre-trained word vecs: 'GoogleNews-vectors-negative300.bin'
- description data: 'alldata.csv'

Returns:
- 'description_vectors.csv' = 300 dimensional 'semantic' vector for each description. One line per item.
- 'description_ids.csv' = IDs corresponding to the semantic vectors. One line per item.

'''

from nltk.corpus import stopwords
import pandas as pd
import nltk
import gensim

# Load pretrained w2v embeddings, trained on GoogleNews
fn = 'GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(fn, binary=True)

# Load stop words
nltk.download()

# Load descriptions
mydata = pd.read_csv('alldata.csv')

d_vecs = []
id_vecs = []
for i in range(0,len(mydata)):

	# Split into lines, use lower case only
    myd = mydata.description[i]
    myd = myd.lower().split()
    
    # Remove stop words
    words = [w for w in myd if not w in stopwords.words("english")]
    
    # For each word in a description, look up w2v embedding. Sum all embeddings for each description.
    mymat = []
    for word in words:
        if word in model.vocab:
            mymat.append(model[word])
    d_vecs.append(sum(mymat))
    id_vecs.append(mydata.id[i])


# Generate and save files with description vectors and corresponding IDs 
d_df = pd.DataFrame(d_vecs)
id_df = pd.DataFrame(id_vecs)
d_df.to_csv('description_vectors.csv')
id_df.to_csv('description_ids.csv')
