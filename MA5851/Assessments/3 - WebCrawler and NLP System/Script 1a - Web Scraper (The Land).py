from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import pandas as pd
import time

firefox_options = Options()
firefox_options.add_argument("--incognito")  # browser operates in private/incognito mode
firefox_options.headless = True  # crawler runs without a visible browser
driver = webdriver.Firefox(options=firefox_options)

startTime = time.time()
# navigate to desired webpage
url = "https://www.theland.com.au/"
driver.get(url)
time.sleep(2)

news_link = driver.find_element_by_css_selector('a.nav-primary__link[href = "/news/"]').get_attribute("href")
driver.get(news_link)
time.sleep(2)

horses_link = driver.find_element_by_css_selector('a.nav-secondary__link[href = "/news/horses/"]').get_attribute("href")
driver.get(horses_link)
time.sleep(2)

# fetch all the links
page_name = driver.title
print("Fetching links...")

stories = driver.find_elements_by_css_selector('div.row.two-block--horizontal')

list_pn = [driver.title for story in stories]
list_h = [story.find_element_by_css_selector('h4.zone-story__headline').text for story in stories]
list_s = [story.find_element_by_css_selector('div.zone-story__summary').text for story in stories]
list_links = [story.find_element_by_css_selector('a.zone-story__clickzone').get_attribute('href') for story in stories]

num_links = len(list_links)

while num_links < 100:
    try:
        next_button = driver.find_element_by_css_selector('a.button.noborder[rel="next"]')
    except:
        print("Finished fetching links.")
        break
    else:
        next_button.click()
        time.sleep(2)

        stories = driver.find_elements_by_css_selector('div.row.two-block--horizontal')

        page_name = [driver.title for story in stories]
        heading = [story.find_element_by_css_selector('h4.zone-story__headline').text for story in stories]
        summary = [story.find_element_by_css_selector('div.zone-story__summary').text for story in stories]
        new_link = [story.find_element_by_css_selector('a.zone-story__clickzone').get_attribute("href") for story in
                    stories]

        list_pn.extend(page_name)
        list_h.extend(heading)
        list_s.extend(summary)
        list_links.extend(new_link)

        num_links = len(list_links)

series_pn = pd.Series(list_pn)
series_h = pd.Series(list_h)
series_s = pd.Series(list_s)
series_links = pd.Series(list_links)

data_dict = {'page_name': series_pn,
             'headline': series_h,
             'summary': series_s,
             'hyperlink': series_links}

links_df = pd.DataFrame(data_dict)

# fetch the content from the stories
list_h2 = []
list_a = []
list_at = []
list_ts = []
list_ts2 = []
list_se = []
list_ic = []

for link in list_links:
    print("Fetching content from " + link)
    driver.get(link)
    time.sleep(2)

    heading = driver.find_element_by_css_selector('h1.story-title.story-header__headline').text

    try:
        author = driver.find_element_by_css_selector('span.story-header__author-name').text
    except:
        author = ''
    finally:
        list_a.append(author)

    try:
        author_twitter = driver.find_element_by_css_selector('span.story-header__author-twitter').text
    except:
        author_twitter = ''
    finally:
        list_at.append(author_twitter)

    pub_date = driver.find_element_by_css_selector('time.story-header__date-pub').get_attribute('datetime')
    pub_date2 = driver.find_element_by_css_selector('time.story-header__date-pub').text
    section = driver.find_element_by_css_selector('div.story-section').text

    try:
        image_caption = driver.find_element_by_css_selector('p.lead-image__caption').text
    except:
        image_caption = ''
    finally:
        list_ic.append(image_caption)

    list_h2.append(heading)
    list_ts.append(pub_date)
    list_ts2.append(pub_date2)
    list_se.append(section)

print("Finished! No more content to extract.")

driver.quit()

executionTime = (time.time() - startTime)
print('Execution time to get content: ' + str(round(executionTime, 2)) + ' seconds')

series_h2 = pd.Series(list_h2)
series_a = pd.Series(list_a)
series_at = pd.Series(list_at)
series_ts = pd.Series(list_ts)
series_ts2 = pd.Series(list_ts2)
series_se = pd.Series(list_se)
series_ic = pd.Series(list_ic)

data_dict = {'headline2': series_h2,
             'author': series_a,
             'twitter_handle': series_at,
             'timestamp': series_ts,
             'timestamp_alt': series_ts2,
             'section': series_se,
             'image_caption': series_ic}

content_df = pd.DataFrame(data_dict)

# join DataFrames together
horses_df = links_df.merge(content_df, left_index=True, right_index=True)
horses_df.to_csv("horses_df.csv")
