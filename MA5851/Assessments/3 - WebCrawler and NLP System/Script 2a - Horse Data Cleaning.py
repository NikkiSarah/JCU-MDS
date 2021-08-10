import pandas as pd

horse_data = pd.read_csv("horses_df.csv", index_col=0)
horse_data.rename(columns={'author': 'contributor_name',
                           'section': 'category'}, inplace=True)

bbc_data = pd.read_csv("Part 1 - Raw scraped data.csv", index_col=0)

horse_data = horse_data[['page_name', 'headline', 'summary', 'hyperlink', 'headline2', 'contributor_name',
                         'timestamp', 'category']]

horse_data.page_name = horse_data.page_name.str.replace("Horses", "Equestrian", regex=False)
horse_data.page_name = horse_data.page_name.str.replace("|", "-", regex=False)
horse_data.page_name = horse_data.page_name.str.replace(" - NSW", "", regex=False)

data = bbc_data.append(horse_data)
data.to_csv("Part 1 - Scraped data.csv")
