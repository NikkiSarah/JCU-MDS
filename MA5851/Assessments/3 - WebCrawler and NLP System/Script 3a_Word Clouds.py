# import packages
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import en_core_web_sm
import regex as re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# read in the data
data = pd.read_csv("Part 2 - Cleaned Data.csv", index_col=0)
data.reset_index(inplace=True, drop=True)


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

headline_text = ' '.join('' if pd.isna(story) else story for story in bow.headline)
print("There are {} words in the combination of all headlines.".format(len(headline_text)))

wordcloud_headline = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                               background_color='white', contour_width=3).generate(headline_text)
plt.imshow(wordcloud_headline, interpolation='bilinear')
plt.axis("off")

headline2_text = ' '.join('' if pd.isna(story) else story for story in bow.headline2)
print("There are {} words in the combination of all headline2s.".format(len(headline2_text)))

wordcloud_headline2 = WordCloud(max_font_size=30, max_words=100000, random_state=2021, scale=2,
                                background_color='white', contour_width=3).generate(headline2_text)
plt.imshow(wordcloud_headline2, interpolation='bilinear')
plt.axis("off")

summary_text = ' '.join('' if pd.isna(story) else story for story in bow.summary)
print("There are {} words in the combination of all summaries.".format(len(summary_text)))

wordcloud_summary = WordCloud(max_font_size=30, max_words=200000, random_state=2021, scale=2,
                              background_color='white', contour_width=3).generate(summary_text)
plt.imshow(wordcloud_summary, interpolation='bilinear')
plt.axis("off")

summary2_text = ' '.join('' if pd.isna(story) else story for story in bow.summary2)
print("There are {} words in the combination of all summary2s.".format(len(summary2_text)))

wordcloud_summary2 = WordCloud(max_font_size=30, max_words=200000, random_state=2021, scale=2,
                               background_color='white', contour_width=3).generate(summary2_text)
plt.imshow(wordcloud_summary2, interpolation='bilinear')
plt.axis("off")

category_text = ' '.join('' if pd.isna(story) else story for story in bow.category)
print("There are {} words in the combination of all categories.".format(len(category_text)))

wordcloud_category = WordCloud(max_font_size=30, random_state=2021, scale=2, background_color='white',
                               contour_width=3).generate(category_text)
plt.imshow(wordcloud_category, interpolation='bilinear')
plt.axis("off")
