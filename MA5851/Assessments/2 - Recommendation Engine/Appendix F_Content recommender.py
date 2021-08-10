# Part 1: Import, pre-process data and split into training and test sets
# import required modules
import en_core_web_sm
import numpy as np
import os
import pandas as pd
import regex as re
import time

from nltk import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# import data from previous script
os.getcwd()
os.chdir(r"D:\Libraries\Documents\Data Science\JCU_MDS\2021_Masterclass 1\Assignment 2")


def import_and_split_data():
    """Imports a csv file from the working directory, extracts all courses with only a single reading, splits the
    data into a training set (80%) and testing set(20%), stratified by the course_id, adds the courses with only a
    single appearance back into the training set and returns three dataframes: df, train and test for use in subsequent
    functions."""

    df = pd.read_csv('Part 2_ontology readings data.csv')
    df = df.drop(columns=['Unnamed: 0', 'CLUSTER'])

    # merge 'UNI' and 'COURSE NAME' to create a unique "user_id" for each combination
    df['COURSE_ID'] = df['UNI'].astype(str) + '_' + df['COURSE_NAME']

    # remove all rows with no metadata
    df = df[df['TITLE'].notna()]

    # remove any titles occurring only once
    title_counts = df['TITLE'].value_counts(ascending=False).reset_index()
    title_counts.columns = ['TITLE', 'TITLE_COUNT']

    df = pd.merge(df, title_counts, on=['TITLE'], how='left')
    df_sub = df[df['TITLE_COUNT'] > 1]

    prop_courses_retained = df_sub['COURSE_ID'].nunique() / df['COURSE_ID'].nunique() * 100
    prop_titles_retained = df_sub['TITLE'].nunique() / df['TITLE'].nunique() * 100

    # count the number of times each course_id appears
    unique_counts = df_sub['COURSE_ID'].value_counts(ascending=True).reset_index()
    unique_counts.columns = ['COURSE_ID', 'COURSE_COUNT']

    df_sub = pd.merge(df_sub, unique_counts, on=['COURSE_ID'], how='left')

    # subset courses into different dataframes based on number of appearances
    single_courses_df = df_sub[df_sub['COURSE_COUNT'] == 1]
    multiple_courses_df = df_sub[df_sub['COURSE_COUNT'] > 1]

    # split dataframe into train and test sets stratified by 'COURSE_ID'
    # append all courses with only one appearance to the training set
    # train-test set split not adjusted as very small number of rows affected
    train, test = train_test_split(multiple_courses_df, test_size=0.2, random_state=2021,
                                   stratify=multiple_courses_df['COURSE_ID'], shuffle=True)

    train = train.append(single_courses_df)

    return train, test, df


train, test, df = import_and_split_data()


# write the files to csv for later use
# train.to_csv("Part 3_content train.csv")
# test.to_csv("Part 3_content test.csv")
# df.to_csv("Part 3_content data.csv")


def preprocess_data(data):
    """Takes in a dataframe and processes the contents suitable for input into a vectoriser. Output is the
    processed dataframe."""
    data = data.copy()
    bow = pd.DataFrame()

    def preprocess_text(text):
        """Takes in a text column from a dataframe and processes the contents into a bag-of-words form. That is, it
        converts words to lowercase, removes punctuation and stopwords (with the spacy stopword list), and stems
        (using the Snowball stemmer) and lemmatises (using the Wordnet lemmatiser) applicable attributes."""
        nlp = en_core_web_sm.load()
        spacy_stopwords = nlp.Defaults.stop_words
        lemmatizer = WordNetLemmatizer()
        stemmer = SnowballStemmer(language='english')

        nona = text.fillna('')
        lower = nona.apply(lambda x: x.lower())
        punc = lower.apply(lambda x: re.sub(r'[^\w\s]', '', x))
        digits = punc.apply(lambda x: re.sub(r'[0-9]', '', x))
        letters = digits.apply(lambda x: re.sub(r'\b\w\b', '', x))
        ws = letters.apply(lambda x: x.strip())
        ws2 = ws.apply(lambda x: re.sub(" +", ' ', x))
        tokens = ws2.apply(lambda x: word_tokenize(x) if x != '' else x)
        stopwords = tokens.apply(lambda x: [word for word in x if word not in spacy_stopwords] if x != '' else x)
        lemmatised = stopwords.apply(lambda x: [lemmatizer.lemmatize(word) for word in x] if x != '' else x)
        stemmed = lemmatised.apply(lambda x: [stemmer.stem(word) for word in x] if x != '' else x)
        return stemmed

    bow['TOPIC'] = data['TOPIC'].fillna('')
    bow['TOPIC'] = bow['TOPIC'].apply(lambda x: re.sub(r'[--]', ' ', x))

    bow['COURSE_NAME'] = preprocess_text(data['COURSE_NAME'])
    bow['TITLE'] = preprocess_text(data['TITLE'])
    bow['AUTHOR'] = preprocess_text(data['AUTHOR'])
    bow['PUBLISHER'] = preprocess_text(data['PUBLISHER'])
    bow['TOPIC'] = preprocess_text(data['TOPIC'])
    bow['FOE'] = preprocess_text(data['FOE'])

    bow['FORMAT'] = data['FORMAT'].apply(lambda x: x.lower())
    bow['FORMAT'] = bow['FORMAT'].apply(lambda x: re.sub(r'[^\w]', '', x))
    bow['FORMAT'] = bow['FORMAT'].apply(lambda x: [x])

    bow['LANGUAGE'] = data['LANGUAGE'].fillna('')
    bow['LANGUAGE'] = bow['LANGUAGE'].apply(lambda x: [x] if x != '' else x)

    # combine desired attributes into a single string (the "product")
    bow['FEATURES'] = ''
    bow['FOE_ORIG'] = data['FOE']
    bow['TITLE_ORIG'] = data['TITLE']
    columns = ['TITLE', 'AUTHOR', 'PUBLISHER', 'FORMAT', 'LANGUAGE', 'TOPIC']

    for index, row in bow.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['FEATURES'] = words

    bow['FEATURES'] = bow['FEATURES'].apply(lambda x: re.sub(" +", ' ', x))

    # append 'COURSE_ID' (the "user")
    bow['COURSE_ID'] = data['COURSE_ID']

    # extract into a separate dataframe and reset the index
    bow_final = bow[['FOE_ORIG', 'COURSE_ID', 'COURSE_NAME', 'TITLE_ORIG', 'FEATURES']]
    bow_final['COURSE_NAME'] = bow['COURSE_NAME'].apply(lambda x: ' '.join(x) + ' ')

    bow_final = bow_final.reset_index()
    bow_final.rename(columns={'index': 'ORIGINAL_INDEX'}, inplace=True)

    indices = pd.Series(bow_final.index)

    return bow_final, indices


bow_train, indices_train = preprocess_data(train)
bow_test, indices_test = preprocess_data(test)


# Part 2: Construct the recommender, run it over the entire training dataset and compute performance metrics
def return_course_id(uni, course_name):
    course_id = str(uni) + "_" + course_name
    return course_id


uni = 2
course_name = "Policing And Crime Prevention"
course_id = return_course_id(uni=uni, course_name=course_name)


# function to train the recommender
def train_content_recommender(course_id, num_recs=10, train_data=bow_train, train=train):
    """Takes in four arguments: (in order) a course_id, number of desired recommendations to be
    returned (with a default of 10), the pre-processed training data returned from the preprocess_data function
    (with a default name of bow_train), and the initial training data created by the train_test_split function (with
    a default name of train). Constructs a tf-idf vectoriser and computes the cosine similarity for all titles with
    respect to the input course name over a selected feature set. Outputs a clean set of recommendations as a
    dataframe suitable for presentation to the user, a raw dataframe that can be used for model evaluation and testing
    and the vectoriser used to train the model."""

    # course_id = "2_Policing And Crime Prevention"
    # num_recs = 10
    # train_data = bow_train

    # reduce corpus to just those texts in the same field of education and reset the index
    #    foe = train_data.loc[train_data['COURSE_ID'] == course_id, 'FOE_ORIG'].iloc[0]
    #    sub_train = train_data[train_data['FOE_ORIG'] == foe].reset_index()
    #    sub_train.rename(columns={'index': 'ORIGINAL_SUB_INDEX'}, inplace=True)

    # reduce the corpus to just those texts in the same general field of education
    # train_data['FOE'] = train_data['FOE_ORIG'].apply(lambda x: re.sub(" &", "", x))
    # train_data['FOE'] = train_data['FOE_ORIG'].apply(lambda x: re.sub(",", "", x))
    # foe = train_data.loc[train_data['COURSE_ID'] == course_id, 'FOE'].iloc[0]

    # import numpy as np
    # train_data['SAME_FOE'] = train_data.apply(lambda x: np.any([word in foe.split(' ')
    #                                                            for word in x['FOE'].split(' ')]),
    #                                          axis = 1)
    # sub_train = train_data[train_data['SAME_FOE']==True]
    # sub_train.reset_index(inplace=True, drop=True)

    # append the pre-processed version of the course name to the top of the feature column as the reference
    # search term for the algorithm
    proc_course_name = pd.Series(train_data[train_data['COURSE_ID'] == course_id].iloc[0, 3])
    features = train_data['FEATURES']
    features = pd.DataFrame(proc_course_name.append(features)).reset_index(drop=True)
    features.rename(columns={0: 'FEATURES'}, inplace=True)

    # create and train the tf-idf vectoriser and compute the cosine similarity matrix
    tfidf_vectoriser = TfidfVectorizer()
    tfidf_matrix = tfidf_vectoriser.fit_transform(features['FEATURES'])
    tfidf_matrix = tfidf_matrix.astype('float32')
    # cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    # apply SVD to reduce the number of features to be trained (and hopefully expedite the training process)
    # explains around 70% of the variance in the data so suitable enough
    content_svd = TruncatedSVD(n_components=1000)
    latent_matrix = content_svd.fit_transform(tfidf_matrix)

    # import matplotlib.pyplot as plt
    # explained = content_svd.explained_variance_ratio_.cumsum()
    # plt.plot(explained, '-', ms = 16, color = "blue")
    # plt.xlabel('Singular value components', fontsize=12)
    # plt.ylabel('Cumulative percent of variance', fontsize =12)

    n = 1000
    latent_train_matrix = pd.DataFrame(latent_matrix[:, 0:n])
    cosine_sim = cosine_similarity(latent_train_matrix[0:1], latent_train_matrix)

    # obtain indices and SIMILARITY_SCOREs and order by descending score
    recommended_titles = []
    idx = indices_train.index[0]
    similarity_scores = pd.Series(cosine_sim[idx])
    similarity_scores = pd.DataFrame(similarity_scores.sort_index())[1:]
    similarity_scores.reset_index(inplace=True, drop=True)
    similarity_scores = similarity_scores.iloc[:, 0].sort_values(ascending=False)

    all_indices = list(similarity_scores.index)

    # obtain the titles of the readings
    for i in all_indices:
        recommended_titles.append(list(train_data['TITLE_ORIG'])[i])

    # combine the columns above and bring in additional metadata columns
    content_recs = pd.DataFrame()
    content_recs['TEXT_ID'] = all_indices
    content_recs['TITLE'] = recommended_titles
    content_recs['SIMILARITY_SCORE'] = list(round(similarity_scores * 100, 2))

    content_recs = pd.merge(content_recs, train_data[['ORIGINAL_INDEX']],
                            how='left', left_on='TEXT_ID', right_index=True)

    content_recs = pd.merge(content_recs, train[['COURSE_ID', 'FORMAT', 'PUB_DATE', 'AUTHOR', 'PUBLISHER',
                                                 'IDENTIFIER', 'TOPIC', 'DESCRIPTION']],
                            how='left', left_on='ORIGINAL_INDEX', right_index=True)

    # create a copy for evaluation purposes and drop duplicates based on 'TITLE'
    all_content_recs = content_recs
    all_content_recs = all_content_recs.drop_duplicates(['TITLE'])
    all_content_recs = all_content_recs.head(num_recs)

    # drop rows for readings already part of the course
    # remove unnecessary columns
    # remove subsequent instances of duplicated readings (i.e. those recommended by more than one course)
    # replace 'nan's with empty strings for better presentation
    content_recs = content_recs[content_recs['COURSE_ID'] != course_id]
    content_recs = content_recs.drop(['TEXT_ID', 'ORIGINAL_INDEX', 'COURSE_ID'], axis=1)

    content_recs = content_recs[content_recs.columns[[0, 2, 3, 4, 5, 6, 7, 1]]]
    content_recs = content_recs.drop_duplicates(['TITLE'])
    content_recs = content_recs.fillna('')
    content_recs = content_recs.head(num_recs)

    return all_content_recs, content_recs, tfidf_vectoriser, content_svd


startTime_train = time.time()

# test function execution on a single course_id (approx 32 seconds)
course_id = "2_Policing And Crime Prevention"
num_recs = 20
all_content_train_recs, content_train_recs, tfidf_vectoriser, content_svd = \
    train_content_recommender(course_id, num_recs)

executionTime_train = (time.time() - startTime_train)
print('Execution time to train the model: ' + str(round(executionTime_train, 2)) + ' seconds')

# recommender system performance evaluation
# extract all titles actually used by each course_id
course_title_train_df = pd.DataFrame()
course_title_train_df = train[['COURSE_ID', 'TITLE', 'COURSE_COUNT']].reset_index()
course_title_train_df.rename(columns={'index': 'TITLE_ID'}, inplace=True)

train_actuals = course_title_train_df.groupby('COURSE_ID', as_index=False)['TITLE_ID'] \
    .agg({'ACTUALS': (lambda x: list(set(x)))})
train_actuals['UNI'] = train_actuals['COURSE_ID'].apply(lambda x: x[0])
train_actuals['COURSE_NAME'] = train_actuals['COURSE_ID'].apply(lambda x: x[2:])

# WARNING: This section takes approx 24 hours to run if looped over all 2,896 course_ids
# take 900 random course_ids from the test set to train the model (this takes approx 8 hours to run)
course_ids = pd.Series(test['COURSE_ID'].unique()).sample(900)

startTime = time.time()

c_id = list(course_ids)

sim_score_list = []
orig_idx_list = []
id_list = []

for id in c_id:
    all_recs, recs, tfidf_vectoriser, content_svd = train_content_recommender(id, 20, bow_train, train)
    # K set at 20 so can calculate performance metrics for different numbers of K if have time (e.g. 10, 15, 20)
    sim_score = all_recs['SIMILARITY_SCORE']
    orig_idx = all_recs['ORIGINAL_INDEX']
    user_id = id
    sim_score_list.append(sim_score)
    orig_idx_list.append(orig_idx)
    id_list.append(user_id)

executionTime_trainAll = (time.time() - startTime)
print('Execution time to make predictions for specified train set: ' +
      str(round(executionTime_trainAll, 2)) + ' seconds')

# extract relevant variables from the output lists to calculate precision, recall, and F1 metrics for each course
train_ids_df = pd.DataFrame(columns=['USER_ID', 'ORIG_IDX', 'SIM_SCORES'])

train_ids_df['USER_ID'], train_ids_df['ORIG_IDX'], train_ids_df['SIM_SCORES'] \
    = id_list, orig_idx_list, sim_score_list

train_ids_df['ORIG_IDX'] = train_ids_df['ORIG_IDX'].apply(lambda x: x.tolist())
train_ids_df['SIM_SCORES'] = train_ids_df['SIM_SCORES'].apply(lambda x: x.tolist())

train_metrics_df = pd.merge(train_ids_df, train_actuals, how='left',
                            left_on='USER_ID', right_on='COURSE_ID')
train_metrics_df = train_metrics_df.iloc[:, [0, 1, 2, 4]]
train_metrics_df.rename(columns={'ORIG_IDX': 'RECS'}, inplace=True)

num_recommended_items_K = 20  # total number of recommended items
num_relevant_items = train_metrics_df['ACTUALS'].apply(lambda x: len(x))

train_metrics_df['NUM_RELEVANT_ITEMS'] = train_metrics_df['ACTUALS'].apply(lambda x: len(x))
train_metrics_df['IRRELEVANT_RECS'] = [list(set(x) - set(y)) for (x, y)
                                       in zip(train_metrics_df['RECS'], train_metrics_df['ACTUALS'])]
train_metrics_df['NUM_CORRECT_RECS'] = train_metrics_df['IRRELEVANT_RECS']. \
    apply(lambda x: num_recommended_items_K - len(x))
train_metrics_df['PRECISION@K'] = train_metrics_df['NUM_CORRECT_RECS'] / num_recommended_items_K
train_metrics_df['RECALL@K'] = train_metrics_df['NUM_CORRECT_RECS'] / train_metrics_df['NUM_RELEVANT_ITEMS']
train_metrics_df['F1@K'] = 2 * ((train_metrics_df['PRECISION@K'] * train_metrics_df['RECALL@K'])
                                / (train_metrics_df['PRECISION@K'] + train_metrics_df['RECALL@K']))
train_metrics_df['AVG_SIM_SCORE@K'] = train_metrics_df['SIM_SCORES'].apply(lambda x: np.mean(x))

# write the results to csv file
train_metrics_df.to_csv("Part 3_content train metrics.csv")

# Part 3: Construct the recommender to be run over the entire test dataset and compute performance metrics
uni = 2
course_name = "Policing And Crime Prevention"
course_id = return_course_id(uni=uni, course_name=course_name)


# function to test the recommender on unseen data
# NOTE: input course_id(s) MUST be the same as the what was used to train the model
def test_content_recommender(course_id, num_recs=10, test_data=bow_test, test=test,
                             vectoriser=tfidf_vectoriser, svd=content_svd):
    """Takes in six arguments: (in order) a course_id, number of desired recommendations to be returned (with a
    default of 10), the pre-processed test data returned from the preprocess_data function (with a default name of
    bow_test), the initial test data created by the train_test_split function (with a default name of train), and
    the trained tf-idf vectoriser and svd transformer from the train_content_recommender function. Computes the
    cosine similarity for all titles with respect to the input course name over a selected feature set. Outputs a
    clean set of recommendations as a dataframe suitable for presentation to the user and a raw dataframe that can be
    used for model evaluation and testing."""

    # course_id = "2_Policing And Crime Prevention"
    # num_recs = 20
    # test_data = bow_test
    # vectoriser = tfidf_vectoriser
    # svd = content_svd

    # append the pre-processed version of the course name to the top of the feature column as the reference
    # search term for the algorithm
    proc_course_name = pd.Series(test_data[test_data['COURSE_ID'] == course_id].iloc[0, 3])
    features = test_data['FEATURES']
    features = pd.DataFrame(proc_course_name.append(features)).reset_index(drop=True)
    features.rename(columns={0: 'FEATURES'}, inplace=True)

    # apply the tf-idf vectoriser and compute the cosine similarity matrix
    tfidf_matrix = vectoriser.transform(features['FEATURES'])
    tfidf_matrix = tfidf_matrix.astype('float32')
    # cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

    # apply SVD to reduce the number of features
    latent_matrix = svd.transform(tfidf_matrix)

    n = 1000
    latent_test_matrix = pd.DataFrame(latent_matrix[:, 0:n])
    cosine_sim = cosine_similarity(latent_test_matrix[0:1], latent_test_matrix)

    # obtain indices and similarity scores and order by descending score
    recommended_titles = []
    idx = indices_test.index[0]
    similarity_scores = pd.Series(cosine_sim[idx])
    similarity_scores = pd.DataFrame(similarity_scores.sort_index())[1:]
    similarity_scores.reset_index(inplace=True, drop=True)
    similarity_scores = similarity_scores.iloc[:, 0].sort_values(ascending=False)

    all_indices = list(similarity_scores.index)

    # obtain the titles of the readings
    for i in all_indices:
        recommended_titles.append(list(test_data['TITLE_ORIG'])[i])

    # combine the columns above and bring in additional metadata columns
    content_recs = pd.DataFrame()
    content_recs['TEXT_ID'] = all_indices
    content_recs['TITLE'] = recommended_titles
    content_recs['SIMILARITY_SCORE'] = list(round(similarity_scores * 100, 2))

    content_recs = pd.merge(content_recs, test_data[['ORIGINAL_INDEX']],
                          how='left', left_on='TEXT_ID', right_index=True)

    content_recs = pd.merge(content_recs, test[['COURSE_ID', 'FORMAT', 'PUB_DATE', 'AUTHOR', 'PUBLISHER',
                                                'IDENTIFIER', 'TOPIC', 'DESCRIPTION']],
                            how='left', left_on='ORIGINAL_INDEX', right_index=True)

    # create a copy for evaluation purposes and drop duplicates based on 'TITLE'
    all_content_recs = content_recs
    all_content_recs = all_content_recs.drop_duplicates(['TITLE'])
    all_content_recs = all_content_recs.head(num_recs)

    # drop rows for readings already part of the course
    # remove unnecessary columns
    # remove subsequent instances of duplicated readings (i.e. those recommended by more than one course)
    # replace 'nan's with empty strings for better presentation
    content_recs = content_recs[content_recs['COURSE_ID'] != course_id]
    content_recs = content_recs.drop(['TEXT_ID', 'ORIGINAL_INDEX', 'COURSE_ID'], axis=1)

    content_recs = content_recs[content_recs.columns[[0, 2, 3, 4, 5, 6, 7, 1]]]
    content_recs = content_recs.drop_duplicates(['TITLE'])
    content_recs = content_recs.fillna('')
    content_recs = content_recs.head(num_recs)

    return all_content_recs, content_recs


startTime_test = time.time()

# test function execution on a single course_id (approx 1.5 seconds)
course_id = course_id
num_recs = 20
all_content_test_recs, content_test_recs = test_content_recommender(course_id, num_recs)

executionTime_test = (time.time() - startTime_test)
print('Execution time to make predictions on the test set: ' + str(round(executionTime_test, 2)) + ' seconds')

# test set performance evaluation
course_title_test_df = pd.DataFrame()
course_title_test_df = test[['COURSE_ID', 'TITLE', 'COURSE_COUNT']].reset_index()
course_title_test_df.rename(columns={'index': 'TITLE_ID'}, inplace=True)

test_actuals = course_title_test_df.groupby('COURSE_ID', as_index=False)['TITLE_ID'] \
    .agg({'ACTUALS': (lambda x: list(set(x)))})
test_actuals['UNI'] = test_actuals['COURSE_ID'].apply(lambda x: x[0])
test_actuals['COURSE_NAME'] = test_actuals['COURSE_ID'].apply(lambda x: x[2:])

# WARNING: This section takes approx 3 hours to run if looped over all 1,662 course_ids
# use 900 random course_ids from the test set to test the model (defined earlier) (this takes approx 23 min to run)
startTime = time.time()

c_id = list(course_ids)

sim_score_list = []
orig_idx_list = []
id_list = []

for id in c_id:
    all_recs, recs = test_content_recommender(id, 20)
    sim_score = all_recs['SIMILARITY_SCORE']
    orig_idx = all_recs['ORIGINAL_INDEX']
    user_id = id
    sim_score_list.append(sim_score)
    orig_idx_list.append(orig_idx)
    id_list.append(user_id)

executionTime_testAll = (time.time() - startTime)
print('Execution time to make predictions for specified test set: '
      + str(round(executionTime_testAll, 2)) + ' seconds')

# extract relevant variables from the output lists to calculate precision, recall, and F1 metrics for each course
test_ids_df = pd.DataFrame(columns=['USER_ID', 'ORIG_IDX', 'SIM_SCORES'])

test_ids_df['USER_ID'], test_ids_df['ORIG_IDX'], test_ids_df['SIM_SCORES'] = id_list, orig_idx_list, sim_score_list

test_ids_df['ORIG_IDX'] = test_ids_df['ORIG_IDX'].apply(lambda x: x.tolist())
test_ids_df['SIM_SCORES'] = test_ids_df['SIM_SCORES'].apply(lambda x: x.tolist())

test_metrics_df = pd.merge(test_ids_df, test_actuals, how='left', left_on='USER_ID', right_on='COURSE_ID')
test_metrics_df = test_metrics_df.iloc[:, [0, 1, 2, 4]]
test_metrics_df.rename(columns={'ORIG_IDX': 'RECS'}, inplace=True)

num_recommended_items_K = 20  # total number of recommended items
num_relevant_items = test_metrics_df['ACTUALS'].apply(lambda x: len(x))

test_metrics_df['NUM_RELEVANT_ITEMS'] = test_metrics_df['ACTUALS'].apply(lambda x: len(x))
test_metrics_df['IRRELEVANT_RECS'] = [list(set(x) - set(y)) for (x, y)
                                      in zip(test_metrics_df['RECS'], test_metrics_df['ACTUALS'])]
test_metrics_df['NUM_CORRECT_RECS'] = test_metrics_df['IRRELEVANT_RECS']. \
    apply(lambda x: num_recommended_items_K - len(x))
test_metrics_df['PRECISION@K'] = test_metrics_df['NUM_CORRECT_RECS'] / num_recommended_items_K
test_metrics_df['RECALL@K'] = test_metrics_df['NUM_CORRECT_RECS'] / test_metrics_df['NUM_RELEVANT_ITEMS']
test_metrics_df['F1@K'] = 2 * ((test_metrics_df['PRECISION@K'] * test_metrics_df['RECALL@K'])
                               / (test_metrics_df['PRECISION@K'] + test_metrics_df['RECALL@K']))
test_metrics_df['AVG_SIM_SCORE@K'] = test_metrics_df['SIM_SCORES'].apply(lambda x: np.mean(x))

# write the results to csv file
test_metrics_df.to_csv("Part 3_content test metrics.csv")
