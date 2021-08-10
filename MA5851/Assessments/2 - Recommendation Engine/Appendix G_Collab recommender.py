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

# import data from previous script
os.getcwd()
os.chdir(r"D:\Libraries\Documents\Data Science\JCU_MDS\2021_Masterclass 1\Assignment 2")


def import_build_ratings_and_split():
    """Imports a csv file from the working directory, removes all titles appearing only once in the dataset,
    computes a set of implicit ratings for each title missing a rating based on its popularity within each
    university's set of readings, determines which courses appear only once in the dataset, splits the
    data into a training set (80%) and testing set(20%), stratified by the course_id, adds the courses with only a
    single appearance back into the training set and returns three dataframes: df, train and test for use in subsequent
    functions."""

    df = pd.read_csv('Part 2_ontology readings data.csv', index_col=0).drop(columns=['CLUSTER'])

    # merge 'UNI' and 'COURSE NAME' to create a unique "user_id" for each combination
    df['COURSE_ID'] = df['UNI'].astype(str) + '_' + df['COURSE_NAME']

    # remove all rows with no metadata
    df = df[df['TITLE'].notna()]

    # remove any titles occurring only once
    title_counts = df['TITLE'].value_counts(ascending=False).reset_index()
    title_counts.columns = ['TITLE', 'TITLE_COUNT']

    df = pd.merge(df, title_counts, on=['TITLE'], how='left')
    df = df[df['TITLE_COUNT'] > 1]

    # (temporarily) separate data into two dataframes based on the existence of pre-existing ratings
    # (temporarily) drop the ratings columns from the dataframe with no ratings data
    titles_with_explicit_ratings = df[df['NUM_RATINGS'].notna()]
    titles_with_no_ratings = df[df['NUM_RATINGS'].isna()].drop(['AVG_RATING', 'NUM_RATINGS'], axis=1)

    # calculate how many times each title is used by each university
    uni_title_counts = titles_with_no_ratings[['UNI', 'TITLE']].value_counts().reset_index()
    uni_title_counts.rename(columns={0: 'UNI_TITLE_COUNT'}, inplace=True)
    uni_title_counts.sort_values(by=['UNI', 'UNI_TITLE_COUNT', 'TITLE'], ascending=[True, False, True], inplace=True)
    uni_title_counts.reset_index(inplace=True, drop=True)

    titles_with_no_ratings = pd.merge(titles_with_no_ratings, uni_title_counts, how='left', on=['UNI', 'TITLE'])
    titles_with_no_ratings.rename(columns={'UNI_TITLE_COUNT': 'NUM_RATINGS'}, inplace=True)

    titles_with_no_ratings['AVG_RATING'] = round((titles_with_no_ratings['NUM_RATINGS'] /
                                                  titles_with_no_ratings['TITLE_COUNT'] * 5) * 2) / 2

    # ensure that every title has a non-zero ratng
    titles_with_no_ratings['AVG_RATING'] = titles_with_no_ratings['AVG_RATING'].apply(lambda x: x if x > 0 else 1.0)

    df = pd.concat([titles_with_explicit_ratings, titles_with_no_ratings])

    # limit the COURSE_IDs to only those who have rated at least 2 titles
    # otherwise unable to run SVD algorithm and calculate cosine similarity
    ratings_counts = df.groupby('COURSE_ID').size().reset_index().sort_values(by=0, ascending=False)

    retained_course_list = ratings_counts[ratings_counts[0] >= 5]['COURSE_ID'].tolist()
    df_sub = df[df['COURSE_ID'].isin(retained_course_list)]
    prop_courses_retained = len(retained_course_list) / len(ratings_counts['COURSE_ID'].tolist()) * 100
    prop_titles_retained = df_sub['TITLE'].nunique() / df['TITLE'].nunique() * 100

    # split dataframe into train and test sets stratified by 'COURSE_ID'
    train, test = train_test_split(df_sub, test_size=0.2, random_state=2021,
                                   stratify=df_sub['COURSE_ID'], shuffle=True)

    return train, test, df


train, test, df = import_build_ratings_and_split()


# write the files to csv for later use
# train.to_csv("Part 3_collab train.csv")
# test.to_csv("Part 3_collab test.csv")
# df.to_csv("Part 3_collab data.csv")

def preprocess_title_course_name(data):
    """Takes in a dataframe and processes TITLE and COURSE NAME for input into the recommender. Output is a
    dataframe containing the processed features."""
    data = data.copy()
    proc = pd.DataFrame()

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

    proc['COURSE_NAME'] = preprocess_text(data['COURSE_NAME'])
    proc['TITLE'] = preprocess_text(data['TITLE'])

    proc['TITLE_ORIG'] = data['TITLE']
    proc['NAME_ORIG'] = data['COURSE_NAME']

    # append 'COURSE_ID' (the "user")
    proc['COURSE_ID'] = data['COURSE_ID']

    # reset the index and rejoin the words in the processed COURSE_NAME and TITLE columns
    proc['COURSE_NAME'] = proc['COURSE_NAME'].apply(lambda x: ' '.join(x) + ' ')
    proc['TITLE'] = proc['TITLE'].apply(lambda x: ' '.join(x) + ' ')

    proc = proc.reset_index()
    proc.rename(columns={'index': 'ORIGINAL_INDEX'}, inplace=True)

    indices = pd.Series(proc.index)

    return proc, indices


proc_train, indices_train = preprocess_title_course_name(train)
proc_test, indices_test = preprocess_title_course_name(test)


# Part 2: Construct the recommender, run it over the entire training dataset and compute performance metrics
def return_course_id(uni, course_name):
    course_id = str(uni) + "_" + course_name
    return course_id


uni = 3
course_name = "Midwifery Foundations"
course_id = return_course_id(uni=uni, course_name=course_name)


# function to train the recommender
def train_collaborative_recommender(course_id, num_recs=10, train_data=proc_train, train=train):
    """Takes in four arguments: (in order) a course_id, number of desired recommendations to be returned (with a
    default of 10), the pre-processed training data returned from the preprocess_title_course_name (with a default
    name of proc_train), and the initial training data created by the train_test_split function (with a default name
    of train). Pivots the data to create a matrix of every course_id against every title and computes the cosine
    similarity for all titles. Outputs a clean set of recommendations as a dataframe suitable for presentation to
    the user, a raw dataframe that can be used for model evaluation and testing, and the SVD transformer and matrix
    used to train the model."""

    # course_id = "3_Colonising Histories"
    # num_recs = 20
    # train_data = proc_train

    train_titles = pd.DataFrame(train['TITLE'].unique())
    train_titles['TITLE_ID'] = np.arange(train_titles.shape[0])
    train_titles.rename(columns={0: 'TITLE'}, inplace=True)

    train_users = pd.DataFrame(train['COURSE_ID'].unique())
    train_users['USER_ID'] = np.arange(train_users.shape[0])
    train_users.rename(columns={0: 'COURSE_ID'}, inplace=True)

    train = pd.merge(train, train_titles, how='left', on='TITLE')
    train = pd.merge(train, train_users, how='left', on='COURSE_ID')

    # append the pre-processed version of the course name to the top of the titles column as the reference
    # search term for the algorithm
    proc_course_name = pd.Series(train_data[train_data['COURSE_ID'] == course_id].iloc[0, 1])
    titles = train_data['TITLE']
    titles = pd.DataFrame(proc_course_name.append(titles)).reset_index(drop=True)
    titles.rename(columns={0: 'TITLES'}, inplace=True)

    # create the user-product matrix and compute the cosine similarity matrix
    user_prod_matrix = train.pivot(index='USER_ID', columns='TITLE_ID', values='AVG_RATING').fillna(0)

    # apply SVD to reduce the number of features to be trained (500 features explains over 80% of the variance in
    # the data)
    collab_svd = TruncatedSVD(n_components=1000)
    latent_matrix = collab_svd.fit_transform(user_prod_matrix)

    # import matplotlib.pyplot as plt
    # explained = collab_svd.explained_variance_ratio_.cumsum()
    # plt.plot(explained, '.-', ms=16, color='red')
    # plt.xlabel('Singular value components', fontsize=12)
    # plt.ylabel('Cumulative percent of variance', fontsize=12);

    n = 500
    latent_train_matrix = pd.DataFrame(latent_matrix[:, 0:n])
    user_vector_index = train_users.loc[train_users['COURSE_ID'] == course_id]['USER_ID']
    user_vector = np.array(latent_train_matrix.loc[user_vector_index]).reshape(1, -1)

    cosine_sim = cosine_similarity(user_vector, latent_train_matrix)

    # obtain indices and similarity scores and order by descending score
    idx = train_users.index[0]
    similarity_scores = pd.DataFrame(cosine_sim[idx])
    similarity_scores.reset_index(inplace=True)
    similarity_scores.rename(columns={'index': 'USER_ID',
                                      0: 'SIMILARITY_SCORE'}, inplace=True)
    similarity_scores.sort_values(by='SIMILARITY_SCORE', ascending=False, inplace=True)
    similarity_scores = similarity_scores.iloc[1:]
    similarity_scores.reset_index(inplace=True, drop=True)

    all_indices = list(similarity_scores['USER_ID'])

    # obtain the course ids and titles used by each of those course ids, and bring in additional metadata columns
    collab_recs = pd.merge(similarity_scores, train[['USER_ID', 'COURSE_ID', 'COURSE_NAME', 'TITLE', 'FORMAT',
                                                     'PUB_DATE', 'AUTHOR', 'PUBLISHER', 'IDENTIFIER', 'TOPIC',
                                                     'DESCRIPTION', 'TITLE_ID']], how='left', on='USER_ID')
    collab_recs['SIMILARITY_SCORE'] = list(round(collab_recs['SIMILARITY_SCORE'] * 100, 2))

    # create a copy for evaluation purposes and drop duplicates based on 'TITLE'
    all_collab_recs = collab_recs
    all_collab_recs = all_collab_recs.drop_duplicates(['TITLE'])
    all_collab_recs = all_collab_recs.head(num_recs)

    # drop rows for readings already part of the course
    # remove unnecessary columns
    # remove subsequent instances of duplicated readings (i.e. those recommended by more than one course)
    # replace 'nan's with empty strings for better presentation
    input_course_id_readings = (train[train['COURSE_ID'] == course_id]['TITLE']).tolist()

    collab_recs = collab_recs[np.logical_not(collab_recs['TITLE'].isin(input_course_id_readings))]
    collab_recs = collab_recs.drop(['USER_ID', 'COURSE_ID', 'TITLE_ID'], axis=1)

    collab_recs = collab_recs[collab_recs.columns[[2, 3, 4, 5, 6, 7, 8, 1, 0]]]
    collab_recs = collab_recs.drop_duplicates(['TITLE'])
    collab_recs = collab_recs.fillna('')
    collab_recs = collab_recs.head(num_recs)
    collab_recs.reset_index(inplace=True, drop=True)

    return all_collab_recs, collab_recs, collab_svd, user_prod_matrix


startTime_train = time.time()

# test function execution on a single course_id (approx 3 seconds)
course_id = course_id
num_recs = 20
all_collab_train_recs, collab_train_recs, collab_svd, collab_matrix = \
    train_collaborative_recommender(course_id, num_recs)

executionTime_train = (time.time() - startTime_train)
print('Execution time to train the model: ' + str(round(executionTime_train, 2)) + ' seconds')

# recommender system performance evaluation on training data
# extract all titles actually used by each course_id
course_counts = train['COURSE_ID'].value_counts(ascending=True).reset_index()
course_counts.columns = ['COURSE_ID', 'COURSE_COUNT']
train = pd.merge(train, course_counts, on=['COURSE_ID'], how='left')

course_title_train_df = pd.DataFrame()
course_title_train_df = train[['COURSE_ID', 'TITLE', 'COURSE_COUNT']].reset_index()
course_title_train_df.rename(columns={'index': 'TITLE_ID'}, inplace=True)

train_actuals = course_title_train_df.groupby('COURSE_ID', as_index=False)['TITLE_ID'] \
    .agg({'ACTUALS': (lambda x: list(set(x)))})
train_actuals['UNI'] = train_actuals['COURSE_ID'].apply(lambda x: x[0])
train_actuals['COURSE_NAME'] = train_actuals['COURSE_ID'].apply(lambda x: x[2:])

# WARNING: This section takes just over 1 hour (~75 min) to run if looped over all 1,188 input course_ids
course_ids = pd.Series(train['COURSE_ID'].unique())

startTime = time.time()

c_id = list(course_ids)

sim_score_list = []
course_idx_list = []
title_idx_list = []
id_list = []

for id in c_id:
    all_recs, recs, collab_svd, collab_matrix = train_collaborative_recommender(id, 20, proc_train, train)
    # K set at 20 so can calculate performance metrics for different numbers of K if have time (e.g. 10, 15, 20)
    sim_score = all_recs['SIMILARITY_SCORE']
    course_idx = all_recs['USER_ID']
    title_idx = all_recs['TITLE_ID']
    user_id = id
    sim_score_list.append(sim_score)
    course_idx_list.append(course_idx)
    title_idx_list.append(title_idx)
    id_list.append(user_id)

executionTime_trainAll = (time.time() - startTime)
print('Execution time to make predictions for specified train set: ' +
      str(round(executionTime_trainAll, 2)) + ' seconds')

# extract relevant variables from the output lists to calculate precision, recall, and F1 metrics for each user
train_ids_df = pd.DataFrame(columns=['USER_ID', 'COURSE_ID', 'TITLE_ID', 'SIM_SCORES'])

train_ids_df['USER_ID'], train_ids_df['COURSE_ID'], train_ids_df['TITLE_ID'], train_ids_df['SIM_SCORES'] \
    = id_list, course_idx_list, title_idx_list, sim_score_list

train_ids_df['COURSE_ID'] = train_ids_df['COURSE_ID'].apply(lambda x: x.tolist())
train_ids_df['TITLE_ID'] = train_ids_df['TITLE_ID'].apply(lambda x: x.tolist())
train_ids_df['SIM_SCORES'] = train_ids_df['SIM_SCORES'].apply(lambda x: x.tolist())

train_metrics_df = pd.merge(train_ids_df, train_actuals, how='left', left_on='USER_ID', right_on='COURSE_ID')
train_metrics_df = train_metrics_df.iloc[:, [0, 1, 2, 3, 5]]
train_metrics_df.rename(columns={'TITLE_ID': 'RECS',
                                 'COURSE_ID_x': 'COURSE_ID'},
                        inplace=True)

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

# write the results to a csv file
train_metrics_df.to_csv("Part 3_collab train metrics.csv")

# Part 3: Construct the recommender to be run over the entire test dataset and compute performance metrics
uni = 3
course_name = "Midwifery Foundations"
course_id = return_course_id(uni=uni, course_name=course_name)


# function to test the recommender on unseen data
# NOTE: input course_id(s) MUST be the same as the what was used to train the model
def test_collaborative_recommender(course_id, num_recs=10, test_data=proc_test, test=test, matrix=collab_matrix,
                                   svd=collab_svd):
    """Takes in six arguments: (in order) a course_id, number of desired recommendations to be returned (with a
    default of 10), the pre-processed test data returned from the preprocess_data function (with a default name of
    bow_test), the initial test data created by the train_test_split function (with a default name of train), the
    trained svd transformer and matrix from the train_collaborative_recommender function (with default names of
    collab_matrix and collab_svd respectively). Computes the cosine similarity for between all users and uses that to
    construct a list of recommended readings. Outputs a clean set of recommendations as a dataframe suitable for
    presentation to the user and a raw dataframe that can be used for model evaluation and testing."""

    # course_id = "3_Colonising Histories"
    # num_recs = 20
    # test_data = proc_test
    # svd = collab_svd
    # matrix = user_prod_matrix

    test_titles = pd.DataFrame(test['TITLE'].unique())
    test_titles['TITLE_ID'] = np.arange(test_titles.shape[0])
    test_titles.rename(columns={0: 'TITLE'}, inplace=True)

    test_users = pd.DataFrame(test['COURSE_ID'].unique())
    test_users['USER_ID'] = np.arange(test_users.shape[0])
    test_users.rename(columns={0: 'COURSE_ID'}, inplace=True)

    test = pd.merge(test, test_titles, how='left', on='TITLE')
    test = pd.merge(test, test_users, how='left', on='COURSE_ID')

    # append the pre-processed version of the course name to the top of the titles column as the reference
    # search term for the algorithm
    proc_course_name = pd.Series(test_data[test_data['COURSE_ID'] == course_id].iloc[0, 1])
    titles = test_data['TITLE']
    titles = pd.DataFrame(proc_course_name.append(titles)).reset_index(drop=True)
    titles.rename(columns={0: 'TITLES'}, inplace=True)

    # apply SVD to reduce the number of features
    latent_matrix = svd.transform(matrix)

    n = 500
    latent_test_matrix = pd.DataFrame(latent_matrix[:, 0:n])
    user_vector_index = test_users.loc[test_users['COURSE_ID'] == course_id]['USER_ID']
    user_vector = np.array(latent_test_matrix.loc[user_vector_index]).reshape(1, -1)

    cosine_sim = cosine_similarity(user_vector, latent_test_matrix)

    # obtain indices and similarity scores and order by descending score
    idx = test_users.index[0]
    similarity_scores = pd.DataFrame(cosine_sim[idx])
    similarity_scores.reset_index(inplace=True)
    similarity_scores.rename(columns={'index': 'USER_ID',
                                      0: 'SIMILARITY_SCORE'},
                             inplace=True)
    similarity_scores.sort_values(by='SIMILARITY_SCORE', ascending=False, inplace=True)
    similarity_scores = similarity_scores.iloc[1:]
    similarity_scores.reset_index(inplace=True, drop=True)

    all_indices = list(similarity_scores['USER_ID'])

    # obtain the course ids and titles used by each of those course ids, and bring in additional metadata columns
    collab_recs = pd.merge(similarity_scores, test[['USER_ID', 'COURSE_ID', 'COURSE_NAME', 'TITLE', 'FORMAT',
                                                    'PUB_DATE', 'AUTHOR', 'PUBLISHER', 'IDENTIFIER', 'TOPIC',
                                                    'DESCRIPTION', 'TITLE_ID']], how='left', on='USER_ID')
    collab_recs['SIMILARITY_SCORE'] = list(round(collab_recs['SIMILARITY_SCORE'] * 100, 2))

    # create a copy for evaluation purposes and drop duplicates based on 'TITLE'
    all_collab_recs = collab_recs
    all_collab_recs = all_collab_recs.drop_duplicates(['TITLE'])
    all_collab_recs = all_collab_recs.head(num_recs)

    # drop rows for readings already part of the course
    # remove unnecessary columns
    # remove subsequent instances of duplicated readings (i.e. those recommended by more than one course)
    # replace 'nan's with empty strings for better presentation
    input_course_id_readings = (test[test['COURSE_ID'] == course_id]['TITLE']).tolist()

    collab_recs = collab_recs[np.logical_not(collab_recs['TITLE'].isin(input_course_id_readings))]
    collab_recs = collab_recs.drop(['USER_ID', 'COURSE_ID', 'TITLE_ID'], axis=1)

    collab_recs = collab_recs[collab_recs.columns[[2, 3, 4, 5, 6, 7, 8, 1, 0]]]
    collab_recs = collab_recs.drop_duplicates(['TITLE'])
    collab_recs = collab_recs.fillna('')
    collab_recs = collab_recs.head(num_recs)
    collab_recs.reset_index(inplace=True, drop=True)

    return all_collab_recs, collab_recs


startTime_test = time.time()

# test function execution on a single course_id (< 1 second)
course_id = return_course_id(uni=uni, course_name=course_name)
num_recs = 20
all_collab_test_recs, collab_test_recs = test_collaborative_recommender(course_id, num_recs)

executionTime_test = (time.time() - startTime_test)
print('Execution time to test the model: ' + str(round(executionTime_test, 2)) + ' seconds')

# recommender system performance evaluation on test set
# extract all titles actually used by each course_id
course_counts = test['COURSE_ID'].value_counts(ascending=True).reset_index()
course_counts.columns = ['COURSE_ID', 'COURSE_COUNT']
test = pd.merge(test, course_counts, on=['COURSE_ID'], how='left')

course_title_test_df = pd.DataFrame()
course_title_test_df = test[['COURSE_ID', 'TITLE', 'COURSE_COUNT']].reset_index()
course_title_test_df.rename(columns={'index': 'TITLE_ID'}, inplace=True)

test_actuals = course_title_test_df.groupby('COURSE_ID', as_index=False)['TITLE_ID'] \
    .agg({'ACTUALS': (lambda x: list(set(x)))})
test_actuals['UNI'] = test_actuals['COURSE_ID'].apply(lambda x: x[0])
test_actuals['COURSE_NAME'] = test_actuals['COURSE_ID'].apply(lambda x: x[2:])

# WARNING: This section takes approx 19 minutes to run if looped over all 1,188 course ids
course_ids = pd.Series(test['COURSE_ID'].unique())

startTime = time.time()

c_id = list(course_ids)

sim_score_list = []
course_idx_list = []
title_idx_list = []
id_list = []

for id in c_id:
    all_recs, recs = test_collaborative_recommender(id, 20)
    # K set at 20 so can calculate performance metrics for different numbers of K if have time (e.g. 10, 15, 20)
    sim_score = all_recs['SIMILARITY_SCORE']
    course_idx = all_recs['USER_ID']
    title_idx = all_recs['TITLE_ID']
    user_id = id
    sim_score_list.append(sim_score)
    course_idx_list.append(course_idx)
    title_idx_list.append(title_idx)
    id_list.append(user_id)

executionTime_testAll = (time.time() - startTime)
print('Execution time to make predictions for specified test set: ' +
      str(round(executionTime_testAll, 2)) + ' seconds')

# extract relevant variables from the output lists to calculate precision, recall, and F1 metrics for each user
test_ids_df = pd.DataFrame(columns=['USER_ID', 'COURSE_ID', 'TITLE_ID', 'SIM_SCORES'])

test_ids_df['USER_ID'], test_ids_df['COURSE_ID'], test_ids_df['TITLE_ID'], test_ids_df['SIM_SCORES'] \
    = id_list, course_idx_list, title_idx_list, sim_score_list

test_ids_df['COURSE_ID'] = test_ids_df['COURSE_ID'].apply(lambda x: x.tolist())
test_ids_df['TITLE_ID'] = test_ids_df['TITLE_ID'].apply(lambda x: x.tolist())
test_ids_df['SIM_SCORES'] = test_ids_df['SIM_SCORES'].apply(lambda x: x.tolist())

test_metrics_df = pd.merge(test_ids_df, test_actuals, how='left', left_on='USER_ID', right_on='COURSE_ID')
test_metrics_df = test_metrics_df.iloc[:, [0, 1, 2, 3, 5]]
test_metrics_df.rename(columns={'TITLE_ID': 'RECS',
                                'COURSE_ID_x': 'COURSE_ID'},
                       inplace=True)

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
test_metrics_df.to_csv("Part 3_collab test metrics.csv")
