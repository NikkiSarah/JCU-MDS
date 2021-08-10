# import packages
import maya
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read in the data
data = pd.read_csv("Part 1 - Scraped data.csv", index_col=0)

# parse timestamp column into a datetime format and convert to the local timezone
data['timestamp_aus'] = data.timestamp.apply(lambda x: x if pd.isna(x) else maya.parse(x)
                                             .datetime(to_timezone='Australia/Melbourne', naive=False))
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

# drop rows with timestamp older than 2020
data = data[data['timestamp'].dt.year >= 2020]

# drop duplicates based on headline column value and keep only first value
data.sort_values(by='headline', inplace=True)
data.drop_duplicates(subset='headline', inplace=True)

# replace subsection column values with nan if value is the same as section column
data.subsection = data.apply(lambda x: x.subsection if x.subsection != x.section else np.nan, axis=1)

# split page_name in two and drop the column
pattern = " - "
page_name_split = data.page_name.apply(lambda x: x.split(pattern, 1))
data['webpage'] = page_name_split.apply(lambda x: x[0])
data['website'] = page_name_split.apply(lambda x: x[1])
data.drop('page_name', axis=1, inplace=True)

# remove "By" from contributor_name and "." from contributor_team
data.contributor_name = data.contributor_name.str.replace("By ", "", regex=False)
data.contributor_team = data.contributor_team.str.replace(".", "", regex=False)

# split contributor_team by newline and retain only the second half
data.contributor_team = data.contributor_team.fillna('')
data.contributor_team = data.contributor_team.apply(lambda x: x if x == '' else x.split('\n')) \
    .apply(lambda x: x[1] if len(x) > 1 else '')

# count the length of summaries to find those that are too short (i.e. contain useless data)
data[['summary', 'summary2']] = data[['summary', 'summary2']].fillna('')

data['summary_length'] = data.summary.apply(lambda x: len(x))
data = data.sort_values(by="summary_length")
data['summary2_length'] = data.summary2.apply(lambda x: len(x))
data = data.sort_values(by="summary2_length")

# replace any summaries with selected strings with np.nan
data.summary = data.summary.apply(lambda x: np.nan if (x == "." or x == "") else x)

data.summary2 = data.summary2.apply(lambda x: np.nan if (x == "."
                                                         or x == "As"
                                                         or x == "The"
                                                         or x == "Claim:"
                                                         or x == "REPORT:"
                                                         or x == "Read more"
                                                         or x == "The claim:"
                                                         or x == "Read more:"
                                                         or x == "READ MORE:"
                                                         or x == "See more at"
                                                         or x == "WATCH MORE:"
                                                         or x == "Follow us on"
                                                         or x == "The government's"
                                                         or x == "All photographs courtesy"
                                                         or x == "All images are copyrighted."
                                                         or x == "Ceri Oakes - Whitby, Yorkshire"
                                                         or x == "All photos subject to copyright"
                                                         or x == "All images subject to copyright"
                                                         or x == "All photos subject to copyright."
                                                         or x == "All images subject to copyright."
                                                         or x == "All photographs courtesy teNeues."
                                                         or x == "All photographs © Corinne Rozotte."
                                                         or x == "All photographs subject to copyright"
                                                         or x == "Images: Reuters, EPA and Getty Images"
                                                         or x == "All photographs courtesy the artists."
                                                         or x == "All photographs subject to copyright."
                                                         or x == "All photographs courtesy Ordnance Survey"
                                                         or x == "All photographs courtesy Dan Giannopoulos") else x)

# create new 'category' column with 'Reality Check' and 'Photography' replaced by nans
data['category2'] = data.category.replace(to_replace=["Reality Check", "Photography"], value=[np.nan, np.nan])

# drop extra/unnecessary columns
data.drop(['summary_length', 'summary2_length', 'scrape_status', 'timestamp', 'website'], axis=1, inplace=True)

# replace '' with nan
data.replace('', np.nan, inplace=True)

# re-order columns
new_order = [0, 2, 3, 4, 7, 12, 10, 11, 5, 6, 8, 9, 1]
data = data[data.columns[new_order]]

data.to_csv("Part 2 - Cleaned data.csv")
data = pd.read_csv("Part 2 - Cleaned data.csv", index_col=0)
data['timestamp_aus'] = data.timestamp_aus.apply(lambda x: x if pd.isna(x) else maya.parse(x)
                                                 .datetime(to_timezone='Australia/Melbourne', naive=False))

# dataset shape and non-null counts
print("The dataset has {} rows and {} columns.".format(data.shape[0], data.shape[1]))
num_rows = data.shape[0]
num_cols = data.shape[1]

data.info()
data_stats = data.describe(datetime_is_numeric=True)

date_from = min(data.timestamp_aus)
date_to = max(data.timestamp_aus)

# number of rows by different columns
counts_webpage = data.webpage.value_counts()
counts_category = data.category.value_counts()
counts_category2 = data.category2.value_counts()
counts_section = data.section.value_counts()
counts_subsection = data.subsection.value_counts()
counts_contributor_name = data.contributor_name.value_counts()
counts_contributor_team = data.contributor_team.value_counts()

# sort story numbers by webpage name instead of count
counts_webpage_s = counts_webpage.sort_index()

# create custom category labels for the stories-by-date bar chart
timestamp_df = pd.DataFrame(data.loc[:, 'timestamp_aus'])
timestamp_df['year'] = pd.DatetimeIndex(data['timestamp_aus']).year
timestamp_df['month'] = pd.DatetimeIndex(data['timestamp_aus']).month
timestamp_df['year-month'] = timestamp_df.apply(lambda x: str(x.year) + "-" + str(x.month), axis=1)
timestamp_counts = timestamp_df['year-month'].value_counts().reset_index()

dateorder = ['2020-1', '2020-2', '2020-3', '2020-4', '2020-5', '2020-6', '2020-7', '2020-8', '2020-9', '2020-10',
             '2020-11', '2020-12', '2021-1', '2021-2', '2021-3', '2021-4']
timestamp_counts['index'] = pd.Categorical(timestamp_counts['index'], categories=dateorder, ordered=True)
timestamp_counts = timestamp_counts.sort_values(by='index')
timestamp_counts.rename(columns={'index': 'year/month', 'year-month': 'count'}, inplace=True)

# change (reduce) text size
plt.rcParams.update({'font.size': 9})
webpage_labels = ['Asia', 'Australia', 'Business', 'Coronavirus', 'Entertainment\n& Arts', 'Equestrian', 'Health',
                  'In Pictures', 'Long Reads', 'Reality Check', 'Science &\nEnvironment', 'Technology', 'UK', 'World']

fig, axs = plt.subplots(1, 2)
axs[0].bar(webpage_labels, counts_webpage_s, color='teal')
axs[0].set_xlabel('Webpage name')
axs[0].set_ylabel('Number of stories')
axs[0].tick_params(axis='x', labelrotation=45)  # labelsize=8
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].bar(timestamp_counts['year/month'], timestamp_counts['count'], color='teal')
axs[1].set_xlabel('Year-month')
axs[1].set_ylabel('Number of stories')
axs[1].tick_params(axis='x', labelrotation=45)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False);
# plt.show()


# summary stats for  of headline, headline2 and summary string lengths
corpus_data = data.loc[:, ['headline', 'headline2', 'summary', 'summary2']]
corpus_data['headline_string_length'] = corpus_data.headline.apply(lambda x: len(x))
corpus_data['headline2_string_length'] = corpus_data.headline2.apply(lambda x: 0 if pd.isna(x) else len(x))
corpus_data['summary_string_length'] = corpus_data.summary.apply(lambda x: 0 if pd.isna(x) else len(x))
corpus_data['summary2_string_length'] = corpus_data.summary2.apply(lambda x: 0 if pd.isna(x) else len(x))

corpus_stats = corpus_data.loc[:, ['headline_string_length',
                                   'headline2_string_length',
                                   'summary_string_length',
                                   'summary2_string_length']].describe()

# reset plot configurations to default
plt.rcdefaults()

# histograms of headline and summary string lengths
# reduce the text size used in the plots
plt.rcParams.update({'font.size': 9})

fig, axs = plt.subplots(2, 2)
fig.suptitle("Number of Stories by Length of the Headline (top) and Summary (bottom) Columns")
axs[0, 0].hist(corpus_data.headline_string_length, bins=80, edgecolor="black", color="teal")
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].set_xlabel('Number of characters')
axs[0, 0].set_ylabel('Number of stories')
axs[0, 1].hist(corpus_data.summary_string_length, bins=250, edgecolor="black", color="teal")
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].set_xlabel('Number of characters')
axs[0, 1].set_ylabel('Number of stories')
axs[1, 0].hist(corpus_data.headline2_string_length, bins=80, edgecolor="black", color="teal")
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].set_xlabel('Number of characters')
axs[1, 0].set_ylabel('Number of stories')
axs[1, 1].hist(corpus_data.summary2_string_length, bins=250, edgecolor="black", color="teal")
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].set_xlabel('Number of characters')
axs[1, 1].set_ylabel('Number of stories');

# if not working in interactive mode, you need to include the following:
# plt.show()

# reset plot configurations to default / change text size used in plots
# plt.rcdefaults()
# plt.rcParams.update({'font.size':9})

# category data bar charts
fig, ax = plt.subplots()
ax.barh(counts_category.index[:30], counts_category[:30], color='teal')
ax.invert_yaxis()
ax.set_xlabel('Number of stories')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False);
# plt.show()

fig, ax = plt.subplots()
ax.barh(counts_category2.index[:30], counts_category2[:30], color='teal')
ax.invert_yaxis()
ax.set_xlabel('Number of stories')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False);
# plt.show()
