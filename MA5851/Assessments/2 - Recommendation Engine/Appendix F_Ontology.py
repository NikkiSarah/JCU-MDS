# import required modules
# data structure
import os
import numpy as np
import pandas as pd
import regex as re

# corpus processing
import nltk
import en_core_web_sm  # fix to a problem with the way spacy is detecting installed packages
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# clustering and evaluation
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Part 2: Pre-Existing Ratings Data Supplementation
# read in dataset and print out basic info
os.getcwd()
os.chdir(r"D:\Libraries\Documents\Data Science\JCU_MDS\2021_Masterclass 1\Assignment 2")

df = pd.read_csv('Part 1_final OpenRefine data.csv')
print(df.info())
print(df.shape)

# remove '_' in all column names
df.rename(columns={'UNI_ID': 'UNI',
                   '_TITLE': 'TITLE',
                   '_FORMAT': 'FORMAT',
                   '_AUTHOR': 'AUTHOR',
                   '_PUB_DATE': 'PUB_DATE',
                   '_PUBLISHER': 'PUBLISHER',
                   '_IDENTIFIER': 'IDENTIFIER',
                   '_TOPIC': 'TOPIC',
                   '_LANGUAGE': 'LANGUAGE',
                   '_DESCRIPTION': 'DESCRIPTION',
                   '_PAGES': 'PAGES',
                   '_AVG_RATING': 'AVG_RATING',
                   '_NUM_RATINGS': 'NUM_RATINGS'},
          inplace=True)

# sort by 'UNI', 'COURSE_NAME', 'TITLE' and 'TITLE STATUS'
# convert 'TITLE' to all titlecase characters
# drop duplicates based on 'UNI', 'COURSE_NAME' and 'TITLE'
df = df.sort_values(by=['UNI', 'COURSE_NAME', 'TITLE', 'TITLE_STATUS'])
df['TITLE'] = df['TITLE'].str.title()
df.drop_duplicates(subset=['UNI', 'COURSE_NAME', 'TITLE'], keep='first', inplace=True)

# consolidate 'FORMAT' column
df['FORMAT'].replace(['undetermined', 'Text - Indeterminate'], 'NotSpecified', inplace=True)
df['FORMAT'] = df['FORMAT'].fillna('NotSpecified')

# extract unique course names into a new dataframe
names_df = pd.DataFrame(df['COURSE_NAME'].unique(), columns=['COURSE_NAME'])
print("Total number of unique course names: " + str(names_df.shape[0]))

# pre-process the text by converting all words to lower-case, removing punctuation and stopwords, and lemmatising
names_list = names_df['COURSE_NAME'].tolist()


def pre_process_text(text):
    nlp = en_core_web_sm.load()
    spacy_stopwords = nlp.Defaults.stop_words

    new_words = []
    for name in text:
        no_punc = re.sub(r'[^\w\s]', '', name)
        tokens = no_punc.split()
        lower_tokens = [token.lower() for token in tokens]
        filtered_tokens = [token for token in lower_tokens if token not in spacy_stopwords]
        new_words.append(" ".join(filtered_tokens))
    return new_words


corpus = pre_process_text(names_list)


# tag the corpus and extract out all noun words
def tag_and_extract_nouns(corpus):
    tagged_names = nltk.pos_tag_sents(map(word_tokenize, corpus))

    noun_words = []
    for i in tagged_names:
        temp = [j[0] for j in i if j[1].startswith('N')]
        noun_words.append(temp)

    flat_noun_names = []
    for name in noun_words:
        temp = ' '.join(name)
        flat_noun_names.append(temp)

    return flat_noun_names


noun_corpus = tag_and_extract_nouns(corpus)

# create a tf-idf matrix allowing for a mix of uni- and bi-grams and use a k-means clustering algorithm
tfidf_vec1 = TfidfVectorizer(lowercase=False, ngram_range=(1, 1), use_idf=True)
tfidf_X1 = tfidf_vec1.fit_transform(noun_corpus)

tfidf_vec2 = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), use_idf=True)
tfidf_X2 = tfidf_vec2.fit_transform(noun_corpus)

# assess different cluster values by sum of squared distance and average silhouette score
chosen_matrix = tfidf_X1
chosen_vec = tfidf_vec1

k_list = list(range(1, 319, 20))
x_list = []
ssd_list = []
labels_list = []

for k in k_list:
    x = k
    km = KMeans(n_clusters=k, random_state=2021, init='k-means++').fit(chosen_matrix)
    labels = km.predict(chosen_matrix)
    ssd = km.inertia_
    x_list.append(x)
    ssd_list.append(ssd)
    labels_list.append(labels)

score_list = []

for array in labels_list[1:]:
    score = silhouette_score(chosen_matrix, array)
    score_list.append(score)

plt.subplot(1,2,1)
plt.title("Elbow plot at various k")
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.plot(x_list, ssd_list, 'o-');

plt.subplot(1,2,2)
plt.title("Average silhouette score at various k")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.plot(x_list[1:], score_list, 'o-');

# run k-means algorithm with chosen number of clusters and tfidf matrix
chosen_k = 70

km_final = KMeans(n_clusters=chosen_k, random_state=2021, init='k-means++').fit(chosen_matrix)
labels = km_final.predict(chosen_matrix)
centroids = km_final.cluster_centers_
ssd = km_final.inertia_
score = silhouette_score(chosen_matrix, labels)

# extract the top 3 terms from each cluster and assign back to course names
order_centroids = km_final.cluster_centers_.argsort()[:, ::-1]
terms = chosen_vec.get_feature_names()

first_indices = [item for sublist in (order_centroids[:, :1].tolist()) for item in sublist]
first_terms = [terms[idx] for idx in first_indices]

second_indices = [item for sublist in (order_centroids[:, 1:2].tolist()) for item in sublist]
second_terms = [terms[idx] for idx in second_indices]

third_indices = [item for sublist in (order_centroids[:, 2:3].tolist()) for item in sublist]
third_terms = [terms[idx] for idx in third_indices]

top_terms = list(zip(first_terms, second_terms, third_terms))

cluster_df = pd.DataFrame(top_terms)
cluster_df['FOE'] = cluster_df[0] + ", " + cluster_df[1] + " & " + cluster_df[2]
cluster_df = cluster_df.reset_index()
cluster_df['CLUSTER'] = cluster_df['index']

counts = np.unique(labels, return_counts=True)

names_df['CLUSTER'] = labels

names_df = pd.merge(names_df, cluster_df, on=['CLUSTER'], how='left')
names_df = names_df.drop(columns=['index', 0, 1, 2])

# join sub-dataframe back to master dataframe
df = pd.merge(df, names_df, on=['COURSE_NAME'], how='left')

# save as a new csv file
df.to_csv('Part 2_ontology_readings data.csv')
