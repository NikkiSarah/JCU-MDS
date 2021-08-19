# import libraries
import gensim
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import regex as re
import seaborn as sns
import spellchecker
import time

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from spellchecker import SpellChecker
from textblob import TextBlob
from wordcloud import WordCloud

# load the language model needed later
# this method imports and the chosen language model as a python module
# the trf model is used over the sm model as it's more accurate (but slower)
import en_core_web_trf
nlp_trf = en_core_web_trf.load()

# just a setting the author's system needs to plot matplotlib charts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# load the data
vodafone_reviews = pd.read_csv('vodafone_reviews1903.csv')


### Part 1: Very Preliminary EDA ###
# define functions to classify each review into a NPS category
def create_nps_category(row):
    if row.score <= 3:
        category = 'Detractor'
    elif row.score == 4:
        category = 'Passive'
    else:
        category = 'Promoter'
    return category

def create_nps_class(row):
    if row.nps_category == 'Detractor':
        nps_class = -1
    elif row.nps_category == 'Passive':
        nps_class = 0
    else:
        nps_class = 1
    return nps_class

vodafone_reviews['nps_category'] = vodafone_reviews.apply(create_nps_category, axis=1)
vodafone_reviews['nps_class'] = vodafone_reviews.apply(create_nps_class, axis=1)

# calculate the number of reviews by customer rating and NPS category
reviews = pd.value_counts(vodafone_reviews.score.values).sort_index()
reviews2 = pd.value_counts(vodafone_reviews.nps_category.values).sort_index()

# plot the number of reviews by customer rating
plt.rcParams.update({'font.size': 12})

plt.subplots()
plt.bar(x=reviews.index, height=reviews.values, color='#990000')
plt.xlabel('Customer rating')
plt.ylabel('Number of reviews')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

# plot the number of reviews by NPS category
plt.subplots()
plt.bar(x=reviews2.index, height=reviews2.values, color='#990000')
plt.xlabel('NPS category')
plt.ylabel('Number of reviews')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

# combine the title and review columns together
vodafone_reviews['text'] = vodafone_reviews.apply(lambda x: x.title + '. ' + x.review, axis=1)

# calculate character and word lengths of the combined text
vodafone_reviews['text_num_chars'] = vodafone_reviews.text.apply(lambda x: len(x))
vodafone_reviews['text_num_words'] = vodafone_reviews.text.apply(lambda x: len(re.findall(r'\w+', x)))

# obtain some basic descriptive statistics
descriptive_statistics = vodafone_reviews.describe()

# plot the distribution of character and word lengths
plt.subplots()
plt.hist(vodafone_reviews.text_num_chars, bins=100, edgecolor='#E60000', color='#990000')
plt.xlabel('Number of characters')
plt.ylabel('Number of reviews');

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

plt.subplots()
plt.hist(vodafone_reviews.text_num_words, bins=25, edgecolor='#E60000', color='#990000')
plt.xlabel('Number of words')
plt.ylabel('Number of reviews');

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

# define a function to display a kde plot of any possible combination of nps category members
def plot_kde_plot(choice=1):
    if choice == 1:
        data = vodafone_reviews[vodafone_reviews.nps_category.isin(["Promoter", "Detractor"])]

        fig, axs = plt.subplots(1, 2)
        p = sns.kdeplot(ax=axs[0], data=data, x='text_num_chars', hue='nps_category', fill=True,
                        common_norm=False, palette=['#990000', '#4a4d4e'], legend=False)
        p.spines['right'].set_visible(False)
        p.spines['top'].set_visible(False)
        p.set_xlabel("Number of characters")
        # p.legend(labels=["Promoter", "Detractor"], title = "NPS category")

        p2 = sns.kdeplot(ax=axs[1], data=data, x='text_num_words', hue='nps_category', fill=True,
                         common_norm=False, palette=['#990000', '#4a4d4e'], legend=False)
        p2.spines['right'].set_visible(False)
        p2.spines['top'].set_visible(False)
        p2.set_xlabel("Number of words")
        p2.set_ylabel("")
        p2.legend(labels=["Promoter", "Detractor"], title="NPS category")
    elif choice == 2:
        fig, axs = plt.subplots(1, 2)
        p = sns.kdeplot(ax=axs[0], data=vodafone_reviews, x='text_num_chars', hue='nps_category', fill=True,
                        common_norm=False, palette=['#990000', '#4a4d4e', '#007c92'], legend=False)
        p.spines['right'].set_visible(False)
        p.spines['top'].set_visible(False)
        p.set_xlabel("Number of characters")
        # p.legend(labels=["Promoter", "Passive", "Detractor"], title="NPS category")

        p2 = sns.kdeplot(ax=axs[1], data=vodafone_reviews, x='text_num_words', hue='nps_category', fill=True,
                         common_norm=False, palette=['#990000', '#4a4d4e', '#007c92'], legend=False)
        p2.spines['right'].set_visible(False)
        p2.spines['top'].set_visible(False)
        p2.set_xlabel("Number of words")
        p2.set_ylabel("")
        p2.legend(labels=["Promoter", "Passive", "Detractor"], title="NPS category")
    elif choice == 3:
        data = vodafone_reviews[vodafone_reviews.nps_category.isin(["Promoter", "Passive"])]

        fig, axs = plt.subplots(1, 2)
        p = sns.kdeplot(ax=axs[0], data=data, x='text_num_chars', hue='nps_category', fill=True,
                        common_norm=False, palette=['#007c92', '#4a4d4e'], legend=False)
        p.spines['right'].set_visible(False)
        p.spines['top'].set_visible(False)
        p.set_xlabel("Number of characters")
        # p.legend(labels=["Promoter", "Passive"], title="NPS category")

        p2 = sns.kdeplot(ax=axs[1], data=data, x='text_num_words', hue='nps_category', fill=True,
                         common_norm=False, palette=['#007c92', '#4a4d4e'], legend=False)
        p2.spines['right'].set_visible(False)
        p2.spines['top'].set_visible(False)
        p2.set_xlabel("Number of words")
        p2.set_ylabel("")
        p2.legend(labels=["Promoter", "Passive"], title="NPS category")
    elif choice == 4:
        data = vodafone_reviews[vodafone_reviews.nps_category.isin(["Detractor", "Passive"])]

        fig, axs = plt.subplots(1, 2)
        p = sns.kdeplot(ax=axs[0], data=data, x='text_num_chars', hue='nps_category', fill=True,
                        common_norm=False, palette=['#990000', '#007c92'], legend=False)
        p.spines['right'].set_visible(False)
        p.spines['top'].set_visible(False)
        p.set_xlabel("Number of characters")
        # p.legend(labels=["Passive", "Detractor"], title="NPS category")

        p2 = sns.kdeplot(ax=axs[1], data=data, x='text_num_words', hue='nps_category', fill=True,
                         common_norm=False, palette=['#990000', '#007c92'], legend=False)
        p2.spines['right'].set_visible(False)
        p2.spines['top'].set_visible(False)
        p2.set_xlabel("Number of words")
        p2.set_ylabel("")
        p2.legend(labels=["Passive", "Detractor"], title="NPS category")
    else:
        print("Invalid selection.")

plt.rcParams.update({'font.size': 10})
plot_kde_plot(choice=1)

# an alternative representation using seaborn pointplots
plt.rcParams.update({'font.size': 12})

p = sns.pointplot(data=vodafone_reviews, x='score', y='text_num_chars', color='#990000', ci=None)
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Customer rating")
p.set_ylabel("Number of characters");

p2 = sns.pointplot(data=vodafone_reviews, x='nps_category', y='text_num_chars', color='#990000', ci=None)
p2.spines['right'].set_visible(False)
p2.spines['top'].set_visible(False)
p2.set_xlabel("NPS category")
p2.set_ylabel("Number of characters");

p3 = sns.pointplot(data=vodafone_reviews, x='score', y='text_num_words', color='#990000', ci=None)
p3.spines['right'].set_visible(False)
p3.spines['top'].set_visible(False)
p3.set_xlabel("Customer rating")
p3.set_ylabel("Number of words");

p4 = sns.pointplot(data=vodafone_reviews, x='nps_category', y='text_num_words', color='#990000', ci=None)
p4.spines['right'].set_visible(False)
p4.spines['top'].set_visible(False)
p4.set_xlabel("NPS category")
p4.set_ylabel("Number of words");

# calculate and plot the correlation between nps_category, customer rating, and title and review lengths
numeric_features = vodafone_reviews.loc[:, ['score', 'nps_class', 'text_num_chars', 'text_num_words']]
corr = numeric_features.corr()

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots()
mask = np.zeros_like(numeric_features.corr())
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(numeric_features.corr(), mask=mask, ax=ax, annot=True, cmap='Reds')

# save all objects created thus far
def save_created_objects():
    object_names = [reviews, reviews2, descriptive_statistics, vodafone_reviews, numeric_features]
    reviews.to_csv('part1_reviews.csv', header=True)
    reviews2.to_csv('part1_reviews2.csv', header=True, index=True)
    descriptive_statistics.to_csv('part1_descriptive_statistics.csv', header=True, index=True)
    vodafone_reviews.to_csv('part1_vodafone_reviews.csv', header=True)
    numeric_features.to_csv('part1_numeric_features.csv', header=True)

save_created_objects()


### Part 2: Data Cleansing and Normalisation Pipeline ###
def preprocess_review_text():
    # convert the text to lower-case
    vodafone_reviews['lower_text'] = vodafone_reviews.text.str.lower()

    # correct curly apostrophes
    vodafone_reviews.lower_text = vodafone_reviews.lower_text.str.replace("’", "'", regex=False)

    # correct encoding errors
    vodafone_reviews.lower_text = vodafone_reviews.lower_text.str.replace("â€™", "'", regex=False)
    vodafone_reviews.lower_text = vodafone_reviews.lower_text.str.replace("â€“", " ", regex=False)
    vodafone_reviews.lower_text = vodafone_reviews.lower_text.str.replace("\r", " ", regex=False)
    vodafone_reviews.lower_text = vodafone_reviews.lower_text.str.replace("\n", " ", regex=False)

    # create a dictionary of common expansions in the english language
    contractions_dict = {"can't": "can not",
                         "won't": "will not",
                         "don't": "do not",
                         "n't": " not",
                         "'m": " am",
                         "'ll": " will",
                         "'d": " would",
                         "'ve": " have",
                         "'re": " are",
                         "'s": ""}  # 's could be 'is' or could be possessive: it has no expansion

    # expand the contractions and add to dataframe as new variable
    exp_text = []
    for review in vodafone_reviews.lower_text:
        text = []
        for key, value in contractions_dict.items():
            if key in review:
                review = review.replace(key, value)
                text.append(review)
        exp_text.append(review)

    vodafone_reviews['clean_text'] = exp_text

    # remove punctuation and clean up the extra white space between words
    vodafone_reviews.clean_text = vodafone_reviews.clean_text.str.replace('[^\w\s]', ' ', regex=True)
    vodafone_reviews.clean_text = vodafone_reviews.clean_text.apply(lambda x: " ".join(x.split()))

    # create a vocabulary from the clean_text column
    def list_of_words(df, column):
        vocabulary = pd.DataFrame(columns=["words"])
        for i in range(len(df)):
            words = df[column].iloc[i]
            words = words.split(" ")
            vocabulary = vocabulary.append(pd.DataFrame(words, columns=["words"]))

        return vocabulary

    vocabulary = list_of_words(vodafone_reviews, "clean_text")
    vocabulary = vocabulary[vocabulary.words != ""]  # remove empty strings

    # initialise a spellchecker and check the unknown words
    spell = SpellChecker()

    unknown_words = spell.unknown(vocabulary.words.to_list())
    unknown_word_counts = vocabulary[vocabulary.words.isin(unknown_words)].value_counts()

    return vodafone_reviews, vocabulary, unknown_words, unknown_word_counts

def perform_advanced_preprocessing(nlp=nlp_trf):
    # extract parts-of-speech and noun phrases
    words = []
    poss = []
    pos_tags = []
    ner_types = []
    noun_chunks = []
    for review in vodafone_reviews.clean_text:
        word = []
        pos = []
        pos_tag = []
        ner_type = []
        chunk = []
        t = nlp(review)
        for w in t:
            word.append(w.text)
            pos.append(w.pos_)
            pos_tag.append(w.tag_)
            ner_type.append(w.ent_type_)
        words.append(word)
        poss.append(pos)
        pos_tags.append(pos_tag)
        ner_types.append(ner_type)

        for c in t.noun_chunks:
            chunk.append(c.text)
        noun_chunks.append(chunk)

    vodafone_reviews['words'] = words
    vodafone_reviews['pos'] = poss
    vodafone_reviews['pos_tags'] = pos_tags
    vodafone_reviews['ner_types'] = ner_types
    vodafone_reviews['noun_phrases'] = noun_chunks

    # pull out named entities (organisations, money, dates etc)
    ent_texts = []
    ent_labels = []
    for review in nlp.pipe(vodafone_reviews.clean_text):
        ent_text = []
        ent_label = []
        for ent in review.ents:
            ent_text.append(ent.text)
            ent_label.append(ent.label_)
        ent_texts.append(ent_text)
        ent_labels.append(ent_label)

    vodafone_reviews['ent_text'] = ent_texts
    vodafone_reviews['ent_label'] = ent_labels

    # check out spacy's stopword list and modify as necessary
    spacy_stopwords = nlp.Defaults.stop_words  # stopwords are the same irrespective of the English language model used

    # lemmatise all words (removing stopwords, punctuation, white space and numbers in the process)
    # note that pos needs stopwords etc to provide context, so the pos and pos_tag columns should be treated with
    #  caution
    preproc_reviews = []
    preproc_poss = []
    preproc_pos_tags = []
    for review in vodafone_reviews.clean_text:
        reviews = []
        pos = []
        pos_tag = []
        t = nlp(review)
        for w in t:
            if not w.is_stop and not w.is_punct and not w.is_digit and not w.is_space:
                reviews.append(w.lemma_)
                pos.append(w.pos_)
                pos_tag.append(w.tag_)
        preproc_reviews.append(reviews)
        preproc_poss.append(pos)
        preproc_pos_tags.append(pos_tag)

    vodafone_reviews['preproc_text'] = preproc_reviews
    vodafone_reviews['preproc_text_pos'] = preproc_poss
    vodafone_reviews['preproc_text_pos_tag'] = preproc_pos_tags

    # add probable bigrams and trigrams
    bigram_model = gensim.models.Phrases(preproc_reviews)
    bigrams = [bigram_model[review] for review in preproc_reviews]

    trigram_model = gensim.models.Phrases(bigrams)
    trigrams = [trigram_model[review] for review in bigrams]

    vodafone_reviews['preproc_bigrams'] = bigrams
    vodafone_reviews['preproc_trigrams'] = trigrams

    return vodafone_reviews

# run both pre-processing functions
vodafone_reviews, vocabulary, unknown_words, unknown_word_counts = preprocess_review_text()

# caution, this function can take a little while to run, the last execution took 18 minutes
start_time = time.time()
perform_advanced_preprocessing()  # specifying an output isn't necessary in this case, don't ask me why
execution_time = time.time() - start_time
print('Execution time in minutes: ' + str(execution_time / 60))

def process_noun_phrases(nlp=nlp_trf):
    # place 'noun_phrases' at the end
    col_name = "noun_phrases"
    last_col = vodafone_reviews.pop(col_name)
    vodafone_reviews.insert(21, col_name, last_col)

    # convert the phrases back into strings
    vodafone_reviews['noun_phrase_text'] = vodafone_reviews.noun_phrases.apply(lambda x: '.'.join(x) if x != '' else x)
    vodafone_reviews.noun_phrase_text = vodafone_reviews.noun_phrase_text.str.replace(" ", "_", regex=False)
    vodafone_reviews.noun_phrase_text = vodafone_reviews.noun_phrase_text.str.replace(".", " ", regex=False)

    lemma_phrases = []
    preproc_phrases = []
    for review in vodafone_reviews.noun_phrase_text:
        lemma_reviews = []
        reviews = []
        t = nlp(review)
        for w in t:
            if not w.is_stop and not w.is_punct and not w.is_digit and not w.is_space:
                lemma_reviews.append(w.lemma_)
                reviews.append(w.text)
        lemma_phrases.append(lemma_reviews)
        preproc_phrases.append(reviews)

    vodafone_reviews['preproc_phrases'] = preproc_phrases
    vodafone_reviews['preproc_lemma_phrases'] = lemma_phrases

    return vodafone_reviews

# caution, this function can take a little while to run, the last execution took 4 minutes
start_time = time.time()
process_noun_phrases()
execution_time = time.time() - start_time
print('Execution time in minutes: ' + str(execution_time / 60))

# define a function to export all objects created in this section
def save_created_objects2():
    object_names = [vocabulary, unknown_words, unknown_word_counts, vodafone_reviews]

    vocabulary.to_csv('part2_vocabulary.csv', header=True)
    unknown_word_counts.to_csv('part2_unknown_word_counts.csv', header=True, index=True)

    with open('part2_unknown_words', 'wb') as outfile:
        pickle.dump(unknown_words, outfile)
    outfile.close()

    with open('part2_vodafone_reviews', 'wb') as outfile:
        pickle.dump(vodafone_reviews, outfile)
    outfile.close()

save_created_objects2()


### Part 3: More Advanced EDA ###
# count and plot the number of Parts of Speech and Parts of Speech tags for both the original and processed text
def count_pos_tags(col, pos_dictionary=None):
    lol = vodafone_reviews[col]
    flat_list = [item for l in lol for item in l]
    freq = Counter(flat_list)
    df = pd.DataFrame.from_dict(freq, orient='index', columns=["count"])

    df['description'] = df.index.map(pos_dictionary)

    return df

coarse_pos_dict = {'ADJ': 'adjective',
                   'ADP': 'adposition',
                   'ADV': 'adverb',
                   'AUX': 'auxiliary',
                   'CONJ': 'conjunction',
                   'CCONJ': 'coordinating conjunction',
                   'DET': 'determiner',
                   'INTJ': 'interjection',
                   'NOUN': 'noun',
                   'NUM': 'numeral',
                   'PART': 'particle',
                   'PRON': 'pronoun',
                   'PROPN': 'proper noun',
                   'PUNCT': 'punctuation',
                   'SCONJ': 'subordinating conjunction',
                   'SYM': 'symbol',
                   'VERB': 'verb',
                   'X': 'other',
                   'SPACE': 'space'}

fine_pos_dict = {'AFX': 'affix',
                 'JJ': 'adjective',
                 'JJR': 'adjective, comparative',
                 'JJS': 'adjective, superlative',
                 'PDT': 'predeterminer',
                 'PRP$': 'pronoun, possessive',
                 'WDT': 'wh-determiner',
                 'WP$': 'wh-pronoun, possessive',
                 'IN': 'conjunction, subordinating or preposition',
                 'EX': 'existential there',
                 'RB': 'adverb',
                 'RBR': 'adverb, comparative',
                 'RBS': 'adverb, superlative',
                 'WRB': 'wh-adverb',
                 'CC': 'conjunction, coordinating',
                 'DT': 'determiner',
                 'UH': 'interjection',
                 'NN': 'noun, singular or mass',
                 'NNS': 'noun, plural',
                 'WP': 'wh-pronoun, personal',
                 'CD': 'cardinal number',
                 'POS': 'possessive ending',
                 'RP': 'adverb, particle',
                 'TO': 'infinitival to',
                 'PRP': 'pronoun, personal',
                 'NNP': 'noun, proper singular',
                 'NNPS': 'noun, proper plural',
                 '-LRB-': 'left round bracket',
                 '-RRB-': 'right round bracket',
                 ',': 'punctuation mark, comma',
                 ':': 'punctuation mark, colon or ellipsis',
                 '.': 'punctuation mark, sentence closer',
                 '”': 'closing quotation mark',
                 '“”': 'closing quotation mark',
                 '': 'opening quotation mark',
                 'HYPH': 'punctuation mark, hyphen',
                 'LS': 'list item marker',
                 'NFP': 'superfluous punctuation',
                 '#': 'symbol, number sign',
                 '$': 'symbol, currency',
                 'SYM': 'symbol',
                 'BES': 'auxiliary “be”',
                 'HVS': 'forms of “have”',
                 'MD': 'verb, modal auxiliary',
                 'VB': 'verb, base form',
                 'VBD': 'verb, past tense',
                 'VBG': 'verb, gerund or present participle',
                 'VBN': 'verb, past participle',
                 'VBP': 'verb, non-3rd person singular present',
                 'VBZ': 'verb, 3rd person singular present',
                 'ADD': 'email',
                 'FW': 'foreign word',
                 'GW': 'additional word in multi-word expression',
                 'XX': 'unknown',
                 '_SP': 'space',
                 'NIL': 'missing tag'}

coarse_pos_orig_df = count_pos_tags("pos", coarse_pos_dict)
coarse_pos_preproc_df = count_pos_tags("preproc_text_pos", coarse_pos_dict)
coarse_pos_df = pd.concat([coarse_pos_orig_df, coarse_pos_preproc_df], axis=1, join='outer').iloc[:, [0,2,1]]
coarse_pos_df.columns = ['original_text', 'preproc_text', 'description']
coarse_pos_df.fillna(0, inplace=True)

fine_pos_orig_df = count_pos_tags("pos_tags", fine_pos_dict)
fine_pos_preproc_df = count_pos_tags("preproc_text_pos_tag", fine_pos_dict)
fine_pos_df = pd.concat([fine_pos_orig_df, fine_pos_preproc_df], axis=1, join='outer').iloc[:, [0,2,1]]
fine_pos_df.columns = ['original_text', 'preproc_text', 'description']
fine_pos_df.fillna(0, inplace=True)

plt.rcParams.update({'font.size': 10})
data = coarse_pos_df.sort_values(by=['original_text'], ascending=False)
p = sns.barplot(x='description', y='original_text', data=data, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Coarse PoS tag")
p.set_ylabel("Frequency")
p.set_xticklabels(data.description, rotation=45, ha='right', fontsize=9);

data = coarse_pos_df.sort_values(by=['preproc_text'], ascending=False)
p = sns.barplot(x='description', y='preproc_text', data=data, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Coarse PoS tag")
p.set_ylabel("Frequency")
p.set_xticklabels(data.description, rotation=45, ha='right', fontsize=9);

plt.rcParams.update({'font.size': 10})
data = fine_pos_df.sort_values(by=['original_text'], ascending=False).head(30)
p = sns.barplot(x='description', y='original_text', data=data, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Fine PoS tag")
p.set_ylabel("Frequency")
p.set_xticklabels(data.description, rotation=45, ha='right', fontsize=9);

data = fine_pos_df.sort_values(by=['preproc_text'], ascending=False).head(30)
p = sns.barplot(x='description', y='preproc_text', data=data, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Fine PoS tag")
p.set_ylabel("Frequency")
p.set_xticklabels(data.description, rotation=45, ha='right', fontsize=9);

# count and plot the number of entities detected
def count_ents(col):
    lol = vodafone_reviews[col]
    flat_list = [item for l in lol for item in l]
    freq = Counter(flat_list)
    df = pd.DataFrame.from_dict(freq, orient='index', columns=["count"])

    return df

ent_text_df = count_ents("ent_text").sort_values(by=['count'], ascending=False)
ent_label_df = count_ents("ent_label").sort_values(by=['count'], ascending=False)

plt.rcParams.update({'font.size': 12})
data = ent_text_df.reset_index().sort_values(by=['count'], ascending=False)[1:31]
p = sns.barplot(x='index', y='count', data=data, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Named entity")
p.set_ylabel("Frequency")
p.set_xticklabels(data['index'], rotation=45, ha='right', fontsize=10);

data = ent_label_df.reset_index().sort_values(by=['count'], ascending=False)
p = sns.barplot(x='index', y='count', data=data, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Named entity type")
p.set_ylabel("Frequency")
p.set_xticklabels(data['index'], rotation=45, ha='right', fontsize=10);

# define a function to create text corpora from selected columns
def create_review_corpora():
    review_corpus = vodafone_reviews.loc[:, ['score', 'nps_category', 'preproc_text',
                                             'preproc_bigrams', 'preproc_trigrams', 'preproc_lemma_phrases']]
    review_corpus['text_strings'] = review_corpus.preproc_text.apply(lambda x: ' '.join(x) if x != '' else x)
    review_corpus['bigram_strings'] = review_corpus.preproc_bigrams.apply(lambda x: ' '.join(x) if x != '' else x)
    review_corpus['trigram_strings'] = review_corpus.preproc_trigrams.apply(lambda x: ' '.join(x) if x != '' else x)
    review_corpus['noun_strings'] = review_corpus.preproc_lemma_phrases.apply(lambda x: ' '.join(x) if x != '' else x)

    promoter_corpus = review_corpus[review_corpus.nps_category == 'Promoter']
    passive_corpus = review_corpus[review_corpus.nps_category == 'Passive']
    detractor_corpus = review_corpus[review_corpus.nps_category == 'Detractor']

    return review_corpus, promoter_corpus, passive_corpus, detractor_corpus

# define a function to create the text format required for generating wordclouds
def generate_wordcloud_text(col_name='noun_strings'):
    all_text = ' '.join('' if pd.isna(review) else review for review in review_corpus[col_name])
    promoter_text = ' '.join('' if pd.isna(review) else review for review in promoter_corpus[col_name])
    passive_text = ' '.join('' if pd.isna(review) else review for review in passive_corpus[col_name])
    detractor_text = ' '.join('' if pd.isna(review) else review for review in detractor_corpus[col_name])

    return all_text, promoter_text, passive_text, detractor_text

review_corpus, promoter_corpus, passive_corpus, detractor_corpus = create_review_corpora()
all_text, promoter_text, passive_text, detractor_text = generate_wordcloud_text()

# overall word cloud
wc_stopwords = ['vodafone', 'vodaphone']
wordcloud_reviews = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2, background_color='white',
                              contour_width=3, stopwords=set(wc_stopwords), colormap='inferno').generate(all_text)
plt.subplots(figsize=(8, 4))
plt.imshow(wordcloud_reviews)
plt.axis("off");

# promoter word cloud
wc_stopwords = ['vodafone']

wordcloud_promoter_reviews = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                                       background_color='white', contour_width=3, stopwords=set(wc_stopwords),
                                       colormap='winter').generate(promoter_text)
plt.subplots(figsize=(8, 4))
plt.imshow(wordcloud_promoter_reviews)
plt.axis("off");

# passive word cloud
wordcloud_passive_reviews = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                                      background_color='white', contour_width=3, stopwords=set(wc_stopwords),
                                      colormap='summer').generate(passive_text)
plt.subplots(figsize=(8, 4))
plt.imshow(wordcloud_passive_reviews)
plt.axis("off");

# detractor word cloud
wordcloud_detractor_reviews = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                                        background_color='white', contour_width=3, stopwords=set(wc_stopwords),
                                        colormap='inferno').generate(detractor_text)
plt.subplots(figsize=(8, 4))
plt.imshow(wordcloud_detractor_reviews)
plt.axis("off");

# check associated word/phrase frequencies
def get_top_n_phrases(corpus, n=None):
    vec = CountVectorizer(stop_words=['vodafone']).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]

all_phrase_counts = get_top_n_phrases(review_corpus.noun_strings, 40)
all_phrase_counts_df = pd.DataFrame(all_phrase_counts, columns=['phrase', 'count'])

promoter_phrase_counts = get_top_n_phrases(promoter_corpus.noun_strings, 40)
promoter_phrase_counts_df = pd.DataFrame(promoter_phrase_counts, columns=['phrase', 'count'])

passive_phrase_counts = get_top_n_phrases(passive_corpus.noun_strings, 40)
passive_phrase_counts_df = pd.DataFrame(passive_phrase_counts, columns=['phrase', 'count'])

detractor_phrase_counts = get_top_n_phrases(detractor_corpus.noun_strings, 40)
detractor_phrase_counts_df = pd.DataFrame(detractor_phrase_counts, columns=['phrase', 'count'])

# plot the dataframes as bar charts
plt.rcParams.update({'font.size': 12})
p = sns.barplot(x='phrase', y='count', data=all_phrase_counts_df, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Noun phrase")
p.set_ylabel("Frequency")
p.set_xticklabels(all_phrase_counts_df.phrase, rotation=45, ha='right', fontsize=9);

p2 = sns.barplot(x='phrase', y='count', data=promoter_phrase_counts_df, palette='rocket')
p2.spines['right'].set_visible(False)
p2.spines['top'].set_visible(False)
p2.set_xlabel("Noun phrase")
p2.set_ylabel("Frequency")
p2.set_xticklabels(promoter_phrase_counts_df.phrase, rotation=45, ha='right', fontsize=9);

p = sns.barplot(x='phrase', y='count', data=passive_phrase_counts_df, palette='rocket')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Noun phrase")
p.set_ylabel("Frequency")
p.set_xticklabels(passive_phrase_counts_df.phrase, rotation=45, ha='right', fontsize=9);

p3 = sns.barplot(x='phrase', y='count', data=detractor_phrase_counts_df, palette='rocket')
p3.spines['right'].set_visible(False)
p3.spines['top'].set_visible(False)
p3.set_xlabel("Noun phrase")
p3.set_ylabel("Frequency")
p3.set_xticklabels(detractor_phrase_counts_df.phrase, rotation=45, ha='right', fontsize=9);

# create new dataframe and calculate sentiment (polarity), subjectivity and classifications
sentiment_df = vodafone_reviews.loc[:, ['score', 'nps_category', 'text', 'clean_text']]
sentiment_df['text_strings'] = review_corpus.loc[:, ['text_strings']]

sentiment_df['text_polarity'] = sentiment_df.text.map(lambda x: TextBlob(x).sentiment.polarity)
sentiment_df['text_subjectivity'] = sentiment_df.text.map(lambda x: TextBlob(x).sentiment.subjectivity)
sentiment_df['clean_polarity'] = sentiment_df.clean_text.map(lambda x: TextBlob(x).sentiment.polarity)
sentiment_df['clean_subjectivity'] = sentiment_df.clean_text.map(lambda x: TextBlob(x).sentiment.subjectivity)

# classify the pattern analyser polarity scores
def classify_polarity(row):
    if row > 0:
        category = 'pos'
    elif row < 0:
        category = 'neg'
    else:
        category = 'neu'

    return category

sentiment_df['text_classification'] = sentiment_df.text_polarity.apply(classify_polarity)
sentiment_df['clean_classification'] = sentiment_df.clean_polarity.apply(classify_polarity)

# compare results against customer rating and nps category
rating_cm = pd.crosstab(sentiment_df.score, sentiment_df.text_classification, rownames=['Customer rating'],
                        colnames=['Textblob'])
nps_cm = pd.crosstab(sentiment_df.nps_category, sentiment_df.text_classification, rownames=['NPS category'],
                     colnames=['Textblob'])
print(rating_cm)
print(nps_cm)

p = sns.boxplot(x='score', y='text_polarity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Customer rating")
p.set_ylabel("Polarity");

p = sns.stripplot(x='score', y='text_polarity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Customer rating")
p.set_ylabel("Polarity");

p = sns.boxplot(x='nps_category', y='text_polarity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("NPS category")
p.set_ylabel("Polarity");

p = sns.stripplot(x='nps_category', y='text_polarity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("NPS category")
p.set_ylabel("Polarity");

p = sns.boxplot(x='score', y='text_subjectivity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Customer rating")
p.set_ylabel("Subjectivity");

p = sns.stripplot(x='score', y='text_subjectivity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("Customer rating")
p.set_ylabel("Subjectivity");

p = sns.boxplot(x='nps_category', y='text_subjectivity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("NPS category")
p.set_ylabel("Subjectivity");

p = sns.stripplot(x='nps_category', y='text_subjectivity', data=sentiment_df, palette='flare')
p.spines['right'].set_visible(False)
p.spines['top'].set_visible(False)
p.set_xlabel("NPS category")
p.set_ylabel("Subjectivity");

# create some descriptive statistics
polarity_stats_by_rating = sentiment_df.groupby('score')['text_polarity'].agg([np.mean, np.std, np.min, np.max, np.median])
polarity_stats_by_nps = sentiment_df.groupby('nps_category')['text_polarity'].agg([np.mean, np.std, np.min, np.max, np.median])

subjectivity_stats_by_rating = sentiment_df.groupby('score')['text_subjectivity'].agg([np.mean, np.std, np.min, np.max, np.median])
subjectivity_stats_by_nps = sentiment_df.groupby('nps_category')['text_subjectivity'].agg([np.mean, np.std, np.min, np.max, np.median])

# save all objects created in this section
def save_created_objects3():
    object_names = [review_corpus, promoter_corpus, passive_corpus, detractor_corpus,
                    all_text, promoter_text, passive_text, detractor_text,
                    all_phrase_counts_df, promoter_phrase_counts_df, passive_phrase_counts_df,
                    detractor_phrase_counts_df,
                    vodafone_reviews,
                    sentiment_df, polarity_stats_by_rating, polarity_stats_by_nps,
                    subjectivity_stats_by_rating, subjectivity_stats_by_nps,
                    rating_cm, nps_cm]

    all_phrase_counts_df.to_csv('part4_all_phrase_counts_df.csv', header=True)
    promoter_phrase_counts_df.to_csv('part4_promoter_phrase_counts_df.csv', header=True)
    passive_phrase_counts_df.to_csv('part4_passive_phrase_counts_df.csv', header=True)
    detractor_phrase_counts_df.to_csv('part4_detractor_phrase_counts_df.csv', header=True)
    sentiment_df.to_csv('part4_sentiment_df.csv', header=True)
    polarity_stats_by_rating.to_csv('part4_polarity_stats_by_rating.csv', header=True, index=True)
    polarity_stats_by_nps.to_csv('part4_polarity_stats_by_nps.csv', header=True, index=True)
    subjectivity_stats_by_rating.to_csv('part4_subjectivity_stats_by_rating.csv', header=True, index=True)
    subjectivity_stats_by_nps.to_csv('part4_subjectivity_stats_by_nps.csv', header=True, index=True)
    rating_cm.to_csv('part4_rating_cm.csv', header=True, index=True)
    nps_cm.to_csv('part4_nps_cm.csv', header=True, index=True)

    with open('part3_review_corpus', 'wb') as outfile:
        pickle.dump(review_corpus, outfile)
    outfile.close()

    with open('part3_promoter_corpus', 'wb') as outfile:
        pickle.dump(promoter_corpus, outfile)
    outfile.close()

    with open('part3_passive_corpus', 'wb') as outfile:
        pickle.dump(passive_corpus, outfile)
    outfile.close()

    with open('part3_detractor_corpus', 'wb') as outfile:
        pickle.dump(detractor_corpus, outfile)
    outfile.close()

    with open('part3_all_text', 'wb') as outfile:
        pickle.dump(all_text, outfile)
    outfile.close()

    with open('part3_promoter_text', 'wb') as outfile:
        pickle.dump(promoter_text, outfile)
    outfile.close()

    with open('part3_passive_text', 'wb') as outfile:
        pickle.dump(passive_text, outfile)
    outfile.close()

    with open('part3_detractor_text', 'wb') as outfile:
        pickle.dump(detractor_text, outfile)
    outfile.close()

    with open('part3_vodafone_reviews', 'wb') as outfile:
        pickle.dump(vodafone_reviews, outfile)
    outfile.close()

save_created_objects3()
