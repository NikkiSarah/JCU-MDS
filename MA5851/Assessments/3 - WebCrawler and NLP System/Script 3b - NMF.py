# import packages
# general and visualisation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pre-processing
import en_core_web_sm
import regex as re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# NMF
import gensim
from gensim.models import CoherenceModel
from itertools import combinations
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# read in the data
data = pd.read_csv("Part 2 - Cleaned data.csv", index_col=0)

# create data subset retaining only the desired features
data_sub = data[['headline', 'summary', 'headline2', 'summary2', 'category', 'webpage']]


def preprocess_data_for_nlp(data):
    """Takes in a dataframe and processes the contents into a form suitable for nlp analysis. Output is a
    processed dataframe."""
    data = data.copy()
    bow = pd.DataFrame()

    def preprocess_text(text):
        """Takes in a text column from a dataframe and processes the contents into a bag-of-words form. That is, it
        converts words to lowercase, removes non-alphabetic words and stopwords (with the spacy stopword list) and
        lemmatises (using the Wordnet lemmatiser) applicable attributes."""
        nlp = en_core_web_sm.load()
        spacy_stopwords = nlp.Defaults.stop_words
        lemmatizer = WordNetLemmatizer()

        nona = text.fillna('')
        lower = nona.apply(lambda x: x.lower())
        punc = lower.apply(lambda x: re.sub(r'[^\w\s]', '', x))
        digits = punc.apply(lambda x: re.sub(r'[0-9]', '', x))
        letters = digits.apply(lambda x: re.sub(r'\b\w\b', '', x))
        tokens = letters.apply(lambda x: word_tokenize(x) if x != '' else x)
        stopwords = tokens.apply(lambda x: [word for word in x if word not in spacy_stopwords] if x != '' else x)
        lemmatised = stopwords.apply(lambda x: [lemmatizer.lemmatize(word) for word in x] if x != '' else x)
        strings = lemmatised.apply(lambda x: ' '.join(x) if x != '' else x)

        return strings

    bow['headline'] = preprocess_text(data.headline)
    bow['summary'] = preprocess_text(data.summary)
    bow['headline2'] = preprocess_text(data.headline2)
    bow['summary2'] = preprocess_text(data.summary2)

    bow['category'] = data['category'].apply(lambda x: x if pd.isna(x) else x.lower())
    bow.replace('', np.nan, inplace=True)

    return bow


bow = preprocess_data_for_nlp(data)

# check if any nulls in headline and/or summary columns and remove rows if true
bow_summary = bow[~pd.isna(bow.summary)]
text_summary = bow_summary.summary

# construct a document-feature matrix for the headline and summary text corpora
vectorizer = TfidfVectorizer()
feature_matrix_summary = vectorizer.fit_transform(text_summary)
terms = vectorizer.get_feature_names()

# examine the number of unique features
print(feature_matrix_summary.shape)

# convert the summaries and headlines into a form suitable for input into a word2vec model
# this only has to be done once
# credit for most of the following code must go to Derek Green (https://github.com/derekgreene/topic-model-tutorial)
text_summary_list = text_summary.to_list()
summary_tokens = [[word for word in sublist.split()] for sublist in text_summary_list]

summary_w2v_model = gensim.models.Word2Vec(sentences=summary_tokens, min_count=1, size=500, sg=1)


# define a function to calculate a simple version of the TC-W2V coherence measure for each of the trained models
def calculate_coherence(w2v_model, term_rankings):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations(term_rankings[topic_index], 2):
            pair_scores.append(w2v_model.similarity(pair[0], pair[1]))
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)


# define function to get a list of the top terms for each topic
def get_top_terms(all_terms, H, topic_index, top_n):
    top_indices = np.argsort(H[topic_index, :])[::-1]
    top_terms = []
    for term_index in top_indices[0:top_n]:
        top_terms.append(all_terms[term_index])
    return top_terms

# define a range of topics and hyperparameters to assess
summary_kmin = 5
summary_kmax = 100
summary_step = 10

p_alpha = [0, 0.2, 0.4]
l1_ratio = [0, 0.2, 0.5, 0.7]

# build nmf models for each of these values for the summary corpus
summary_nmf_models = []
for k in range(summary_kmin, summary_kmax+1, summary_step):
    for alpha in p_alpha:
        for ratio in l1_ratio:
            print("Applying NMF to the summary text for k = {}, alpha = {}, ratio = {}...".format(k, alpha, ratio))
            model = NMF(n_components=k, init = 'nndsvd', alpha = alpha, l1_ratio=ratio, random_state=2021)
            W = model.fit_transform(feature_matrix_summary)
            H = model.components_
            summary_nmf_models.append((k, W, H, alpha, ratio))

# process each of the models for the different values of k
summary_results = []
for (k, W, H, alpha, ratio) in summary_nmf_models:
    term_rankings = []
    for topic_index in range(k):
        term_rankings.append(get_top_terms(terms, H, topic_index, 20))
    summary_results.append({
        'k': k,
        'alpha': alpha,
        'ratio': ratio,
        'coherence': calculate_coherence(summary_w2v_model, term_rankings)
    })

summary_results = pd.DataFrame(summary_results)
summary_results_s = summary_results.sort_values(by=['alpha','ratio'])
summary_results_s.reset_index(inplace=True, drop=True)

# create the line plot
ax1 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][:9], linestyle = '--')
ax2 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][40:49])
ax3 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][50:59])
ax4 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][60:69])
ax5 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][70:79])
ax6 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][80:89])
ax7 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][90:99])
ax8 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][100:109])
ax9 = plt.plot(summary_results_s['k'][:9], summary_results_s['coherence'][110:119])
plt.xticks(summary_results_s['k'][:9])
plt.xlabel("Number of topics")
plt.ylabel("Mean coherence score")

# determine the best number of topics for the tuned hyperparameters
summary_kmin = 5
summary_kmax = 150
summary_step = 5

summary_nmf_models = []
for k in range(summary_kmin, summary_kmax+1, summary_step):
    print("Applying NMF to the summary text for k = {}.".format(k))
    model = NMF(n_components=k, init = 'nndsvd', alpha = 0.4, l1_ratio=0, random_state=2021)
    W = model.fit_transform(feature_matrix_summary)
    H = model.components_
    summary_nmf_models.append((k, W, H))

# process each of the models for the different values of k
summary_k_values = []
summary_coherences = []
for (k, W, H) in summary_nmf_models:
    term_rankings = []
    for topic_index in range(k):
        term_rankings.append(get_top_terms(terms, H, topic_index, 20))
    summary_k_values.append(k)
    summary_coherences.append(calculate_coherence(summary_w2v_model, term_rankings))
    print("Corpus = 'summaries', K = %02d: Coherence = %.4f" % (k, summary_coherences[-1]))

plt.plot(summary_k_values, summary_coherences, color='teal')
plt.xticks(summary_k_values)
plt.xlabel("Number of topics")
plt.ylabel("Mean coherence score")
plt.scatter(summary_k_values, summary_coherences, edgecolors='teal')
ymax = max(summary_coherences)
xpos = summary_coherences.index(ymax)
best_k = summary_k_values[xpos]
plt.annotate("k = %d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points")

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

k = 70
W = summary_nmf_models[13][1]
H = summary_nmf_models[13][2]

topics_and_terms = []
for topic_index in range(k):
    top_terms = get_top_terms(terms, H, topic_index, 20)
    str_term = ', '.join(top_terms)
    topics_and_terms.append({'topic': topic_index, 'top terms': str_term})

topics_and_terms = pd.DataFrame(topics_and_terms)
topics_and_terms = topics_and_terms['top terms'].str.split(', ', expand=True).transpose()

# examine which summaries have the strongest link to each topic to help determine labels
nmf_embedding = W
nmf_embedding = (nmf_embedding - nmf_embedding.mean(axis=0)) / nmf_embedding.std(axis=0)

top_idx = pd.DataFrame(np.argsort(nmf_embedding, axis=0)[-20:])

topic_num = []
topic_indexes = []
summary_text = []
all_summary_text = []
for (colname, coldata) in top_idx.iteritems():
    number = colname
    indexes = coldata.values
    topic_num.append(number)
    topic_indexes.append(indexes)
    all_summary_text.append(summary_text)
    summary_text = []
    for idx in indexes:
        text = bow_summary.iloc[idx]['summary']
        summary_text.append(text)

all_summary_text_s = pd.Series(all_summary_text)
all_summary_text_s = all_summary_text_s[1:]

all_summary_text_s2 = all_summary_text_s.apply(pd.Series)
topics_and_summaries = all_summary_text_s2.transpose()
topics_and_summaries = topics_and_summaries.T.reset_index(drop=True).T

# initial ideas for topic labels for first 10 topics
# some topics appear multiple times so may want to re-examine and reduce cluster sizes or adjust hyperparameters
topic_labels = ['Topic 1: photography', 'Topic 2: photography months', 'Topic 3: health', 'Topic 4: cricket',
                'Topic 5: vaccinating the young', 'Topic 6: the year...', 'Topic 7: covid recovery',
                'Topic 8: photography reader theme', 'Topic 9: covid lockdown easing', 'Topic 10: Australian Open']

# repeat the same process as above but for the headlines
# check if any nulls in headline and/or summary columns and remove rows if true
bow_headline = bow[~pd.isna(bow.headline)]
text_headline = bow_headline.headline

# construct a document-feature matrix for the headline and summary text corpora
vectorizer = TfidfVectorizer()
feature_matrix_headline = vectorizer.fit_transform(text_headline)
terms = vectorizer.get_feature_names()

# examine the number of unique features
print(feature_matrix_headline.shape)

# convert the summaries and headlines into a form suitable for input into a word2vec model
# this only has to be done once
text_headline_list = text_headline.to_list()
headline_tokens = [[word for word in sublist.split()] for sublist in text_headline_list]

headline_w2v_model = gensim.models.Word2Vec(sentences=headline_tokens, min_count=1, size=500, sg=1)

# define a range of topics and hyperparameters to assess
headline_kmin = 5
headline_kmax = 100
headline_step = 10

p_alpha = [0, 0.2, 0.4]
l1_ratio = [0, 0.2, 0.5, 0.7]

# build nmf models for each of these values for the headline corpus
headline_nmf_models = []
for k in range(headline_kmin, headline_kmax+1, headline_step):
    for alpha in p_alpha:
        for ratio in l1_ratio:
            print("Applying NMF to the headline text for k = {}, alpha = {}, ratio = {}...".format(k, alpha, ratio))
            model = NMF(n_components=k, init = 'nndsvd', alpha = alpha, l1_ratio=ratio, random_state=2021)
            W = model.fit_transform(feature_matrix_headline)
            H = model.components_
            headline_nmf_models.append((k, W, H, alpha, ratio))

# process each of the models for the different values of k
headline_results = []
for (k, W, H, alpha, ratio) in headline_nmf_models:
    term_rankings = []
    for topic_index in range(k):
        term_rankings.append(get_top_terms(terms, H, topic_index, 20))
    headline_results.append({
        'k': k,
        'alpha': alpha,
        'ratio': ratio,
        'coherence': calculate_coherence(headline_w2v_model, term_rankings)
    })

headline_results = pd.DataFrame(headline_results)
headline_results_s = headline_results.sort_values(by=['alpha','ratio'])
headline_results_s.reset_index(inplace=True, drop=True)

# create the line plot
ax1 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][:9], linestyle = '--')
ax2 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][40:49])
ax3 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][50:59])
ax4 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][60:69])
ax5 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][70:79])
ax6 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][80:89])
ax7 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][90:99])
ax8 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][100:109])
ax9 = plt.plot(headline_results_s['k'][:9], headline_results_s['coherence'][110:119])
plt.xticks(headline_results_s['k'][:9])
plt.xlabel("Number of topics")
plt.ylabel("Mean coherence score")

# determine the best number of topics for the tuned hyperparameters
headline_kmin = 5
headline_kmax = 150
headline_step = 5

headline_nmf_models = []
for k in range(headline_kmin, headline_kmax+1, headline_step):
    print("Applying NMF to the headline text for k = {}.".format(k))
    model = NMF(n_components=k, init = 'nndsvd', alpha = 0.4, l1_ratio=0, random_state=2021)
    W = model.fit_transform(feature_matrix_headline)
    H = model.components_
    headline_nmf_models.append((k, W, H))

# process each of the models for the different values of k
headline_k_values = []
headline_coherences = []
for (k, W, H) in headline_nmf_models:
    term_rankings = []
    for topic_index in range(k):
        term_rankings.append(get_top_terms(terms, H, topic_index, 20))
    headline_k_values.append(k)
    headline_coherences.append(calculate_coherence(headline_w2v_model, term_rankings))
    print("Corpus = 'summaries', K = %02d: Coherence = %.4f" % (k, headline_coherences[-1]))

plt.plot(headline_k_values, headline_coherences, color='teal')
plt.xticks(headline_k_values)
plt.xlabel("Number of topics")
plt.ylabel("Mean coherence score")
plt.scatter(headline_k_values, headline_coherences, edgecolors='teal')
ymax = max(headline_coherences)
xpos = headline_coherences.index(ymax)
best_k = headline_k_values[xpos]
plt.annotate("k = %d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points")

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

k = 10
W = headline_nmf_models[1][1]
H = headline_nmf_models[1][2]

topics_and_terms = []
for topic_index in range(k):
    top_terms = get_top_terms(terms, H, topic_index, 20)
    str_term = ', '.join(top_terms)
    topics_and_terms.append({'topic': topic_index, 'top terms': str_term})

topics_and_terms = pd.DataFrame(topics_and_terms)
topics_and_terms = topics_and_terms['top terms'].str.split(', ', expand=True).transpose()

# examine which summaries have the strongest link to each topic to help determine labels
nmf_embedding = W
nmf_embedding = (nmf_embedding - nmf_embedding.mean(axis=0)) / nmf_embedding.std(axis=0)

top_idx = pd.DataFrame(np.argsort(nmf_embedding, axis=0)[-20:])

topic_num = []
topic_indexes = []
headline_text = []
all_headline_text = []
for (colname, coldata) in top_idx.iteritems():
    number = colname
    indexes = coldata.values
    topic_num.append(number)
    topic_indexes.append(indexes)
    all_headline_text.append(headline_text)
    headline_text = []
    for idx in indexes:
        text = bow_headline.iloc[idx]['headline']
        headline_text.append(text)

all_headline_text_s = pd.Series(all_headline_text)
all_headline_text_s = all_headline_text_s[1:]

all_headline_text_s2 = all_headline_text_s.apply(pd.Series)
topics_and_headlines = all_headline_text_s2.transpose()
topics_and_headlines = topics_and_headlines.T.reset_index(drop=True).T

# initial ideas for topic labels for first 10 topics
# some topics appear multiple times so may want to re-examine and reduce cluster sizes or adjust hyperparameters
headline_topic_labels = ['Topic 1: photography months', 'Topic 2: covid jab', 'Topic 3: covid lockdown',
                         'Topic 4: cricket', 'Topic 5: covid vaccine', 'Topic 6: Prince Phillip death',
                         'Topic 7: UK-EU trade deal', 'Topic 8: Australian Open', 'Topic 9: coronavirus pandemic',
                         'Topic 10: Africa shot']
