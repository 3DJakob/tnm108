d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

my_stop_words={"the","is"}
my_vocabulary={'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
vectorizer=CountVectorizer(stop_words=my_stop_words,vocabulary=my_vocabulary)
smatrix = vectorizer.transform(Z)
matrix = smatrix.todense()
print(matrix)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)
print(matrix)

# print idf values
feature_names = vectorizer.get_feature_names()
import pandas as pd
df_idf=pd.DataFrame(tfidf_transformer.idf_, index=feature_names,columns=["idf_weights"])
# sort ascending
df_idf.sort_values(by=['idf_weights'])

print(df_idf)

# tf-idf scores over the sparse graph
tf_idf_vector = tfidf_transformer.transform(smatrix)
# get tfidf vector for first document
first_document = tf_idf_vector[0] # first document "The sky is blue."
# print the scores
df=pd.DataFrame(first_document.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)
print(df)


# COSINE SIMILARITY

d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)

import math
# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity[0][2])
print(math.degrees(angle_in_radians))
