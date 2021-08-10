# import packages
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import maya
import numpy as np
import pandas as pd

# read in the data
data = pd.read_csv("Part 2 - Cleaned Data.csv", index_col=0)
data.reset_index(inplace=True, drop=True)
data['timestamp_aus'] = data.timestamp_aus.apply(lambda x: x if pd.isna(x) else maya.parse(x)
                                                 .datetime(to_timezone='Australia/Melbourne', naive=False))

# create data subset retaining only the desired features
data_sub = data[['headline', 'summary', 'category', 'webpage']]


# textblob sentiment analysis
def get_textblob_scores(dataframe, column):
    """Function to add the polarity and subjectivity scores for all possible sentiment types to the input dataframe
    given a dataframe and a text column to calculate scores for."""
    dataframe[[f'{column} polarity', f'{column} subjectivity']] = dataframe[column]. \
        apply(lambda x: pd.Series(TextBlob(x).sentiment))
    return dataframe


# get polarity and subjectivity scores for headline and summary columns
# polarity ranges between -1 and 1, where -1 is very negative and 1 is very positive
# subjectivity ranges between 0 and 1, where 0 is very objective and 1 is very subjective
textblob_sentiment = data.copy()
textblob_sentiment[['summary', 'headline2', 'summary2']] = textblob_sentiment[
    ['summary', 'headline2', 'summary2']].fillna('')

textblob_sentiment = get_textblob_scores(textblob_sentiment, 'headline')
textblob_sentiment = get_textblob_scores(textblob_sentiment, 'summary')


# replace values that should be null with np.nan to avoid skewing the results of subsequent analysis
def correct_results(dataframe, column, comp_column):
    dataframe[column] = dataframe.apply(lambda x: x[column] if x[comp_column] != '' else np.nan, axis=1)
    return dataframe


textblob_sentiment = correct_results(textblob_sentiment, 'summary polarity', 'summary')
textblob_sentiment = correct_results(textblob_sentiment, 'summary subjectivity', 'summary')


# classify each headline/summary as positive, neutral or negative based on the compound score
def get_textblob_rating(dataframe, column):
    if pd.isna(dataframe[column]):
        return "N/A"
    elif dataframe[column] > 0.1:
        return 'Positive'
    elif dataframe[column] < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


textblob_sentiment['headline rating'] = textblob_sentiment.apply(lambda x: get_textblob_rating(x, 'headline polarity'),
                                                                 axis=1)
textblob_sentiment['summary rating'] = textblob_sentiment.apply(lambda x: get_textblob_rating(x, 'summary polarity'),
                                                                axis=1)

# number of stories by rating (overall, by webpage, by category)
textblob_rating_counts = pd.DataFrame()
textblob_rating_counts['headline'] = textblob_sentiment['headline rating'].value_counts()
textblob_rating_counts['summary'] = textblob_sentiment['summary rating'].value_counts()

textblob_webpage_rating_counts = pd.DataFrame()
textblob_webpage_rating_counts['headline'] = textblob_sentiment.groupby(['webpage', 'headline rating']).size()
textblob_webpage_rating_counts['summary'] = textblob_sentiment.groupby(['webpage', 'summary rating']).size()
textblob_webpage_rating_counts.reset_index(inplace=True)
textblob_webpage_rating_counts.rename(columns={'headline rating': 'rating'}, inplace=True)

textblob_category_rating_counts = pd.DataFrame()
textblob_category_rating_counts['headline'] = textblob_sentiment.groupby(['category', 'headline rating']).size()
textblob_category_rating_counts['summary'] = textblob_sentiment.groupby(['category', 'summary rating']).size()
textblob_category_rating_counts.reset_index(inplace=True)
textblob_category_rating_counts.rename(columns={'headline rating': 'rating'}, inplace=True)


# standard descriptive stats
textblob_polarity_stats = pd.DataFrame()
textblob_polarity_stats['headline'] = textblob_sentiment['headline polarity'].describe()
textblob_polarity_stats['summary'] = textblob_sentiment['summary polarity'].describe()

# polarity by webpage and category
textblob_agg_polarity_by_webpage = textblob_sentiment.groupby(['webpage']).mean().reset_index()

# create copies of dataframe sorted by descending headline and summary compound sentiment score
textblob_agg_polarity_by_webpage_shd = textblob_agg_polarity_by_webpage.sort_values(by='headline polarity',
                                                                                    ascending=False)
textblob_agg_polarity_by_webpage_ssd = textblob_agg_polarity_by_webpage.sort_values(by='summary polarity',
                                                                                    ascending=False)


textblob_category_counts = textblob_sentiment.category.value_counts()
textblob_polarity_by_category = textblob_sentiment.groupby(['category']).mean().reset_index()

textblob_agg_polarity_by_category = pd.merge(textblob_category_counts, textblob_polarity_by_category,
                                             how='left',
                                             left_index=True, right_on='category')
textblob_agg_polarity_by_category.reset_index(inplace=True, drop=True)
textblob_agg_polarity_by_category.drop('category_y', axis=1, inplace=True)
textblob_agg_polarity_by_category.rename(columns={'category_x': 'num stories'}, inplace=True)

# create copies of dataframe retaining only categories with more than nine stories and sorted by descending headline
# and summary polarity
textblob_agg_polarity_by_category_gt9 = textblob_agg_polarity_by_category\
    [textblob_agg_polarity_by_category['num stories'] > 9]
textblob_agg_polarity_by_category_gt9_shd = textblob_agg_polarity_by_category_gt9.sort_values(by='headline polarity',
                                                                                              ascending=False)
textblob_agg_polarity_by_category_gt9_ssd = textblob_agg_polarity_by_category_gt9.sort_values(by='summary polarity',
                                                                                              ascending=False)

# create copies of dataframe retaining only categories with more than nine stories and sorted by ascending headline
# and summary polarity
textblob_agg_polarity_by_category_gt9_sha = textblob_agg_polarity_by_category_gt9.sort_values(by='headline polarity')
textblob_agg_polarity_by_category_gt9_ssa = textblob_agg_polarity_by_category_gt9.sort_values(by='summary polarity')

# vader sentiment analysis
# download the vader lexicon if it isn't already installed; you will need to import nltk to do so
# nltk.download('vader_lexicon')
analyser = SentimentIntensityAnalyzer()


def get_sentiment(text, analyser, type):
    """Function to get sentiment of a particular type (positive, neutral, negative or compound) from input text."""
    sentiment_score = analyser.polarity_scores(text)
    return sentiment_score[type]


def get_vader_scores(dataframe, column):
    """Function to add the sentiment scores for all possible sentiment types to the input dataframe given a dataframe
    and a text column to calculate sentiment scores for. Uses the previous get_sentiment function."""
    dataframe[f'{column} pos sent score'] = dataframe[column].apply(lambda x: x if pd.isna(x)
        else get_sentiment(x, analyser, 'pos'))
    dataframe[f'{column} neg sent score'] = dataframe[column].apply(lambda x: x if pd.isna(x)
        else get_sentiment(x, analyser, 'neg'))
    dataframe[f'{column} neu sent score'] = dataframe[column].apply(lambda x: x if pd.isna(x)
        else get_sentiment(x, analyser, 'neu'))
    dataframe[f'{column} comp sent score'] = dataframe[column].apply(lambda x: x if pd.isna(x)
        else get_sentiment(x, analyser, 'compound'))
    return dataframe


# get sentiment for headline and summary columns
vader_sentiment = data_sub.copy()
vader_sentiment = get_vader_scores(vader_sentiment, 'headline')
vader_sentiment = get_vader_scores(vader_sentiment, 'summary')


# classify each headline/summary as positive, neutral or negative based on the compound score
def get_rating(dataframe, column):
    if pd.isna(dataframe[column]):
        return "N/A"
    elif dataframe[column] > 0.05:
        return 'Positive'
    elif dataframe[column] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


vader_sentiment['headline rating'] = vader_sentiment.apply(lambda x: get_rating(x, 'headline comp sent score'), axis=1)
vader_sentiment['summary rating'] = vader_sentiment.apply(lambda x: get_rating(x, 'summary comp sent score'), axis=1)

# number of stories by rating (overall, by webpage, by category)
vader_rating_counts = pd.DataFrame()
vader_rating_counts['headline'] = vader_sentiment['headline rating'].value_counts()
vader_rating_counts['summary'] = vader_sentiment['summary rating'].value_counts()

vader_webpage_rating_counts = pd.DataFrame()
vader_webpage_rating_counts['headline'] = vader_sentiment.groupby(['webpage', 'headline rating']).size()
vader_webpage_rating_counts['summary'] = vader_sentiment.groupby(['webpage', 'summary rating']).size()
vader_webpage_rating_counts.reset_index(inplace=True)
vader_webpage_rating_counts.rename(columns={'headline rating': 'rating'}, inplace=True)

vader_category_rating_counts = pd.DataFrame()
vader_category_rating_counts['headline'] = vader_sentiment.groupby(['category', 'headline rating']).size()
vader_category_rating_counts['summary'] = vader_sentiment.groupby(['category', 'summary rating']).size()
vader_category_rating_counts.reset_index(inplace=True)
vader_category_rating_counts.rename(columns={'headline rating': 'rating'}, inplace=True)


# standard descriptive stats
def get_descriptive_stats(dataframe, column, source_dataframe):
    """Function to calculate descriptive statistics for each sentiment score column given an output dataframe,
    column to calculate the statistics for and the source dataframe."""
    dataframe[f'{column} pos'] = source_dataframe[f'{column} pos sent score'].describe()
    dataframe[f'{column} neg'] = source_dataframe[f'{column} neg sent score'].describe()
    dataframe[f'{column} neu'] = source_dataframe[f'{column} neu sent score'].describe()
    dataframe[f'{column} comp'] = source_dataframe[f'{column} comp sent score'].describe()
    return dataframe


vader_sentiment_stats = pd.DataFrame()
vader_sentiment_stats = get_descriptive_stats(vader_sentiment_stats, 'headline', vader_sentiment)
vader_sentiment_stats = get_descriptive_stats(vader_sentiment_stats, 'summary', vader_sentiment)

# sentiment scores by webpage and category
vader_agg_sentiment_by_webpage = vader_sentiment.groupby(['webpage']).mean().reset_index()

vader_category_counts = vader_sentiment.category.value_counts()
vader_sentiment_by_category = vader_sentiment.groupby(['category']).mean().reset_index()

vader_agg_sentiment_by_category = pd.merge(vader_category_counts, vader_sentiment_by_category,
                                           how='left',
                                           left_index=True, right_on='category')
vader_agg_sentiment_by_category.reset_index(inplace=True, drop=True)
vader_agg_sentiment_by_category.drop('category_y', axis=1, inplace=True)
vader_agg_sentiment_by_category.rename(columns={'category_x': 'num stories'}, inplace=True)

# create copies of dataframe sorted by descending headline and summary compound sentiment score
vader_agg_sentiment_by_webpage_shd = vader_agg_sentiment_by_webpage.sort_values(by='headline comp sent score',
                                                                                ascending=False)
vader_agg_sentiment_by_webpage_ssd = vader_agg_sentiment_by_webpage.sort_values(by='summary comp sent score',
                                                                                ascending=False)

# create copies of dataframe retaining only categories with more than nine stories and sorted by descending headline
# and summary polarity
vader_agg_sentiment_by_category_gt9 = vader_agg_sentiment_by_category\
    [vader_agg_sentiment_by_category['num stories'] > 9]
vader_agg_sentiment_by_category_gt9_shd = vader_agg_sentiment_by_category_gt9.sort_values(by='headline comp sent score',
                                                                                          ascending=False)
vader_agg_sentiment_by_category_gt9_ssd = vader_agg_sentiment_by_category_gt9.sort_values(by='summary comp sent score',
                                                                                          ascending=False)

# create copies of dataframe retaining only categories with more than nine stories and sorted by ascending headline
# and summary polarity
vader_agg_sentiment_by_category_gt9_sha = vader_agg_sentiment_by_category_gt9.sort_values(by='headline comp sent score')
vader_agg_sentiment_by_category_gt9_ssa = vader_agg_sentiment_by_category_gt9.sort_values(by='summary comp sent score')

# visualisations
# comparison by average score
plt.rcParams.update({'font.size': 9})
vader_summary = vader_sentiment_stats.loc['mean', ['headline comp', 'summary comp']]
textblob_summary = textblob_polarity_stats.loc['mean']

fig, axs = plt.subplots(1, 2)
axs[0].bar(textblob_summary.index, textblob_summary, color='teal')
axs[0].set_xlabel('TextBlob')
axs[0].set_ylabel('Mean polarity')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].bar(textblob_summary.index, vader_summary, color='maroon')
axs[1].set_xlabel('VADER')
axs[1].set_ylabel('Mean compound sentiment score')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False);

# comparison by webpage
webpage_labels1 = ['In Pictures', 'Equestrian', 'Coronavirus', 'Science &\nEnvironment', 'UK', 'Entertainment\n& Arts',
                   'Australia', 'Business', 'Asia', 'Health', 'World', 'Long Reads', 'Technology', 'Reality Check']
webpage_labels2 = ['Equestrian', 'In Pictures', 'Entertainment\n& Arts', 'Coronavirus', 'Science &\nEnvironment',
                   'Business', 'Health', 'Australia', 'UK', 'Technology', 'Long Reads', 'Reality Check', 'Asia',
                   'World']
webpage_labels3 = ['In Pictures', 'Equestrian', 'Entertainment\n& Arts', 'Science &\nEnvironment', 'Health',
                   'Coronavirus', 'Technology', 'UK', 'Long Reads', 'Australia', 'Business', 'Asia', 'World',
                   'Reality Check']
webpage_labels4 = ['Equestrian', 'In Pictures', 'Entertainment\n& Arts', 'Coronavirus', 'Science &\nEnvironment',
                   'Technology', 'Business', 'Health', 'UK', 'Australia', 'Long Reads', 'Asia', 'Reality Check',
                   'World']

plt.rcParams.update({'font.size': 9})
fig, axs = plt.subplots(2, 2)
axs[0,0].bar(webpage_labels1, textblob_agg_polarity_by_webpage_shd['headline polarity'],
              color=(textblob_agg_polarity_by_webpage_shd['headline polarity'] > 0)
              .map({True: 'teal', False: 'maroon'}))
axs[0,0].set_xlabel('Webpage name')
axs[0,0].set_ylabel('Mean headline polarity')
axs[0,0].tick_params(axis='x', labelrotation=45)  # labelsize=8
axs[0,0].spines['top'].set_visible(False)
axs[0,0].spines['right'].set_visible(False)
axs[0,1].bar(webpage_labels2, vader_agg_sentiment_by_webpage_shd['headline comp sent score'],
             color=(vader_agg_sentiment_by_webpage_shd['headline comp sent score'] > 0).map({True: 'teal',
                                                                                             False: 'maroon'}))
axs[0,1].set_xlabel('Webpage name')
axs[0,1].set_ylabel('Mean headline compound sentiment')
axs[0,1].tick_params(axis='x', labelrotation=45)  # labelsize=8
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)
axs[1,0].bar(webpage_labels3, textblob_agg_polarity_by_webpage_ssd['summary polarity'],
             color=(textblob_agg_polarity_by_webpage_ssd['summary polarity'] > 0).map({True: 'teal', False: 'maroon'}))
axs[1,0].set_xlabel('Webpage name')
axs[1,0].set_ylabel('Mean summary polarity')
axs[1,0].tick_params(axis='x', labelrotation=45)
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['right'].set_visible(False)
axs[1,1].bar(webpage_labels4, vader_agg_sentiment_by_webpage_ssd['summary comp sent score'],
             color=(vader_agg_sentiment_by_webpage_ssd['summary comp sent score'] > 0).map({True: 'teal',
                                                                                            False: 'maroon'}))
axs[1,1].set_xlabel('Webpage name')
axs[1,1].set_ylabel('Mean summary compound sentiment score')
axs[1,1].tick_params(axis='x', labelrotation=45)
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['right'].set_visible(False);
# plt.show()

# comparison by category
# bar chart of sentiment by category (for the top-30 categories by number of stories)
plt.rcParams.update({'font.size': 9})
fig, axs = plt.subplots(1, 2)
axs[0].barh(textblob_agg_polarity_by_category.category[:30],
            textblob_agg_polarity_by_category['headline polarity'][:30],
            color=(textblob_agg_polarity_by_category['headline polarity'] > 0).map({True: 'teal', False: 'maroon'}))
axs[0].invert_yaxis()
axs[0].set_xlabel('Mean headline polarity score')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].barh(vader_agg_sentiment_by_category.category[:30],
            vader_agg_sentiment_by_category['headline comp sent score'][:30],
            color=(vader_agg_sentiment_by_category['headline comp sent score'] > 0).map({True: 'teal',
                                                                                         False: 'maroon'}))
axs[1].invert_yaxis()
axs[1].set_xlabel('Mean headline compound sentiment score')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False);
# plt.show()

# bar chart of sentiment by category (for the top-20 most-negative and most-positive categories)
plt.rcParams.update({'font.size': 9})
fig, axs = plt.subplots(2, 2)
axs[0,0].barh(textblob_agg_polarity_by_category_gt9_sha.category[:20],
            textblob_agg_polarity_by_category_gt9_sha['headline polarity'][:20],
            color='maroon')
axs[0,0].invert_yaxis()
axs[0,0].set_xlabel('Mean headline polarity')
axs[0,0].spines['top'].set_visible(False)
axs[0,0].spines['right'].set_visible(False);
axs[0,1].barh(vader_agg_sentiment_by_category_gt9_sha.category[:20],
            vader_agg_sentiment_by_category_gt9_sha['headline comp sent score'][:20],
            color='maroon')
axs[0,1].invert_yaxis()
axs[0,1].set_xlabel('Mean headline compound sentiment score')
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)
axs[1,0].barh(textblob_agg_polarity_by_category_gt9_shd.category[:20],
            textblob_agg_polarity_by_category_gt9_shd['headline polarity'][:20],
            color='teal')
axs[1,0].invert_yaxis()
axs[1,0].set_xlabel('Mean headline polarity')
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['right'].set_visible(False);
axs[1,1].barh(vader_agg_sentiment_by_category_gt9_shd.category[:20],
            vader_agg_sentiment_by_category_gt9_shd['headline comp sent score'][:20],
            color='teal')
axs[1,1].invert_yaxis()
axs[1,1].set_xlabel('Mean headline compound sentiment score')
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['right'].set_visible(False)
# plt.show()
