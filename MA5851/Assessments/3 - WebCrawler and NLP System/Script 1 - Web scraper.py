# import necessary libraries
# if the user does not already have selenium installed, then they should visit
# 'https://www.selenium.dev/documentation/en/' for installation instructions and webdriver requirement directions
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import pandas as pd
import time

# establish webdriver and set desired options. Any driver webdriver can be used but the default is Firefox
firefox_options = Options()
firefox_options.add_argument("--incognito")  # browser operates in private/incognito mode
#firefox_options.headless = True  # crawler runs without a visible browser
driver = webdriver.Firefox(options=firefox_options)

# an extra line of code is required if the webdriver is named anything other than 'driver'
# driver = ...

# list of BBC News urls for the scraper to visit
url_list = ['https://www.bbc.com/news/world',
            'https://www.bbc.com/news/world/asia',
            'https://www.bbc.com/news/world/australia',
            'https://www.bbc.com/news/uk',
            'https://www.bbc.com/news/technology',
            'https://www.bbc.com/news/science_and_environment',
            'https://www.bbc.com/news/reality_check',
            'https://www.bbc.com/news/in_pictures',
            'https://www.bbc.com/news/health',
            'https://www.bbc.com/news/entertainment_and_arts',
            'https://www.bbc.com/news/coronavirus',
            'https://www.bbc.com/news/business']


# function to scrape the content available on the "Latest News" section
def scrape_preliminary_content(url_list, num_pages=39):
    # initialise lists
    list_pn = []
    list_h = []
    list_s = []
    list_hl = []

    for url in url_list:
        # navigate to each url in turn and halt execution for two seconds
        driver.get(url)
        time.sleep(2)

        # locate and extract the webpage name and current page number
        page_name = driver.title
        page_current = driver.find_element_by_css_selector(
            'span.lx-pagination__page-number.qa-pagination-current-page-number').text
        # pages_total = driver.find_element_by_css_selector
        # ('span.lx-pagination__page-number.qa-pagination-total-page-number').text

        while int(page_current) < num_pages:
            # halt execution for two seconds and then locate the current page number
            time.sleep(2)
            page_current = driver.find_element_by_css_selector(
                'span.lx-pagination__page-number.qa-pagination-current-page-number').text

            print('Preliminary content scrape in progress on page {} of {} of {}'
                  .format(page_current, str(num_pages), page_name[0]))

            # locate all stories on the current webpage
            stories = driver.find_elements_by_css_selector('li.lx-stream__post-container')

            # locate and extract page name, headline, summary/description information and the hyperlink for each story
            page_name = [driver.title for story in stories]
            headline = [story.find_element_by_css_selector('header.lx-stream-post__header').text for story in stories]
            list_pn.extend(page_name)
            list_h.extend(headline)

            for story in stories:
                try:
                    summary = story.find_element_by_css_selector('p.lx-stream-related-story--summary').text
                except:
                    try:
                        summary = story.find_element_by_css_selector('p.lx-media-asset-summary').text
                    except:
                        summary = ''
                finally:
                    list_s.append(summary)

            for story in stories:
                try:
                    hyperlink = story.find_element_by_css_selector('a.qa-heading-link').get_attribute("href")
                except:
                    hyperlink = ''
                finally:
                    list_hl.append(hyperlink)

            # locate the pagination ribbon and then the button to the next page, click it and halt execution for two
            # seconds
            pagination = driver.find_element_by_css_selector('div.lx-pagination__nav')
            next_button = pagination.find_element_by_css_selector('a[rel="next"]')
            next_button.click()
            time.sleep(2)

        else:
            # let the user know that the while loop is finished (i.e. a progress update)
            print('Done! Proceeding to next webpage.')

    # convert the lists into a set of series and then aggregate into a dataframe
    series_hl = pd.Series(list_hl)
    series_pn = pd.Series(list_pn)
    series_h = pd.Series(list_h)
    series_s = pd.Series(list_s)

    data_dict = {'hyperlink': series_hl,
                 'page_name': series_pn,
                 'headline': series_h,
                 'summary': series_s}

    prelim_df = pd.DataFrame(data_dict)

    # sort the data and drop duplicate headlines
    prelim_deduped = prelim_df.sort_values(by='headline')
    prelim_deduped = prelim_deduped.drop_duplicates(subset='headline', keep='first')

    # split the data by the presence of a hyperlink element
    prelim_hyperlinks = prelim_deduped[prelim_deduped.hyperlink != '']
    prelim_no_hyperlinks = prelim_deduped[prelim_deduped.hyperlink == '']

    return prelim_hyperlinks, prelim_no_hyperlinks


# function to combine the main preliminary content extractor with the customised one for the "Long Reads" webpage as
# it had fewer pages from which to scrape content
def concatenate_preliminary_content(url_list):
    print("This could take a while. Hope you're not pressed for time.")

    prelim_hyperlinks, prelim_no_hyperlinks = scrape_preliminary_content(url_list=url_list)
    prelim_hyperlinks_long, prelim_no_hyperlinks_long = scrape_preliminary_content \
        (url_list=['https://www.bbc.com/news/the_reporters'], num_pages=29)

    print("Done! Proceeding to next stage.")

    frames_hyperlinks = [prelim_hyperlinks, prelim_hyperlinks_long]
    frames_no_hyperlinks = [prelim_no_hyperlinks, prelim_no_hyperlinks_long]

    prelim_hyperlink_data = pd.concat(frames_hyperlinks, sort=True)
    prelim_no_hyperlink_data = pd.concat(frames_no_hyperlinks, sort=True)

    hyperlink_list = prelim_hyperlink_data['hyperlink'].tolist()

    prelim_hyperlink_data['hyperlink'].to_csv('Hyperlink list.csv')
    prelim_hyperlink_data.to_csv("Preliminary hyperlink data.csv")
    prelim_no_hyperlink_data.to_csv("Preliminary no hyperlink data.csv")

    return prelim_hyperlink_data, prelim_no_hyperlink_data, hyperlink_list


# function to scrape the content available within each story with a hyperlink. Follows almost exactly the same
# process as scrape_preliminary_content except the scrape uses a list of hyperlinks instead of a list of URLs
def scrape_detailed_content(hyperlink_list):
    list_status = []
    list_h2 = []
    list_s2 = []
    list_ts = []
    list_cn = []
    list_ct = []
    list_c = []
    list_se = []
    list_ss = []

    for hyperlink in hyperlink_list:
        try:
            driver.get(hyperlink)
            time.sleep(3)

            status = "Success"
            print("Full content scrape in progress for link {} of {}"
                  .format(hyperlink_list.index(hyperlink), len(hyperlink_list)))

            try:
                headline2 = driver.find_element_by_id('main-heading').text
            except:
                try:
                    headline2 = driver.find_element_by_id('lx-event-title').text
                except:
                    try:
                        headline2 = driver.find_element_by_css_selector('h1.gel-trafalgar-bold.qa-story-headline').text
                    except:
                        headline2 = ''
            finally:
                list_h2.append(headline2)

            try:
                summary2 = driver.find_element_by_css_selector('div.ssrcss-3z08n3-RichTextContainer.e5tfeyi2').text
            except:
                try:
                    summary2 = driver.find_element_by_css_selector('b.ssrcss-14iz86j-BoldText.e5tfeyi0').text
                except:
                    try:
                        summary2 = driver.find_element_by_css_selector('ol.lx-c-summary-points.gel-long-primer').text
                    except:
                        try:
                            summary2 = driver.find_element_by_css_selector('p.qa-introduction').text
                        except:
                            summary2 = ''
            finally:
                list_s2.append(summary2)

            try:
                timestamp = driver.find_element_by_css_selector('time[data-testid="timestamp"]') \
                    .get_attribute('datetime')
            except:
                try:
                    timestamp = driver.find_element_by_css_selector('time.gs-o-bullet__text.qa-status-date') \
                        .get_attribute('datetime')
                except:
                    timestamp = ''
            finally:
                list_ts.append(timestamp)

            try:
                contributor_name = driver.find_element_by_css_selector('p.ssrcss-1pjc44v-Contributor.e5xb54n0') \
                    .find_element_by_tag_name('strong').text
            except:
                try:
                    contributor_name = driver.find_element_by_css_selector('span.qa-contributor-name').text
                except:
                    contributor_name = ''
            finally:
                list_cn.append(contributor_name)

            try:
                contributor_team = driver.find_element_by_css_selector('p.ssrcss-1pjc44v-Contributor.e5xb54n0').text
            except:
                try:
                    contributor_team = driver.find_element_by_css_selector('span.qa-contributor-title').text
                except:
                    contributor_team = ''
            finally:
                list_ct.append(contributor_team)

            try:
                category = driver.find_element_by_css_selector('a.ssrcss-1yno9a1-StyledLink.ed0g1kj1').text
            except:
                category = ''
            finally:
                list_c.append(category)

            try:
                section = driver.find_elements_by_css_selector('a.ssrcss-zf4gw3-MetadataLink.ecn1o5v1')[0].text
            except:
                try:
                    section = driver.find_element_by_css_selector('a[href = "/sport/"]').text
                except:
                    section = ''
            finally:
                list_se.append(section)

            try:
                subsection = driver.find_elements_by_css_selector('a.ssrcss-zf4gw3-MetadataLink.ecn1o5v1')[1].text
            except:
                subsection = ''
            finally:
                list_ss.append(subsection)

        except:
            status = "Failure"
            continue
        finally:
            list_status.append(status)

    # notify when detailed content scrape has finished
    print('Finished! No more content to scrape.')

    # shut down driver
    driver.quit()

    # convert all lists produced into pandas Series
    series_status = pd.Series(list_status)
    series_h2 = pd.Series(list_h2)
    series_s2 = pd.Series(list_s2)
    series_ts = pd.Series(list_ts)
    series_cn = pd.Series(list_cn)
    series_ct = pd.Series(list_ct)
    series_c = pd.Series(list_c)
    series_se = pd.Series(list_se)
    series_ss = pd.Series(list_ss)

    # combine pandas Series into a DataFrame
    data_dict = {'scrape_status': series_status,
                 'headline2': series_h2,
                 'summary2': series_s2,
                 'timestamp': series_ts,
                 'contributor_name': series_cn,
                 'contributor_team': series_ct,
                 'category': series_c,
                 'section': series_se,
                 'subsection': series_ss}

    det_data = pd.DataFrame(data_dict)

    return det_data


# function to merge the preliminary content with the detailed content and append the stories without hyperlinks
def concatenate_detailed_content(hyperlink_list, prelim_hyperlink_data, prelim_no_hyperlink_data):
    print("This is probably going to take even longer. Go away and come back in a couple of hours.")
    det_data = scrape_detailed_content(hyperlink_list=hyperlink_list)

    # shut down the driver once scrape is complete
    driver.quit()

    prelim_hyperlink_data.reset_index(inplace=True, drop=True)

    joined_data = pd.merge(prelim_hyperlink_data, det_data, how='left', left_index=True, right_index=True)
    bbc_data = joined_data.append(prelim_no_hyperlink_data)

    bbc_data.to_csv("Part 1 - Scraped data.csv")

    return bbc_data


startTime_all = time.time()
startTime_prelim = time.time()
prelim_hyperlink_data, prelim_no_hyperlink_data, hyperlink_list = concatenate_preliminary_content(url_list)
executionTime_prelim = (time.time() - startTime_prelim)
print('Time taken to scrape preliminary content: {} minutes.'.format(str(round(executionTime_prelim, 2) / 60)))

startTime_det = time.time()
bbc_data = concatenate_detailed_content(hyperlink_list, prelim_hyperlink_data, prelim_no_hyperlink_data)
executionTime_det = (time.time() - startTime_det)
print('Time taken to scrape detailed content: {} minutes.'.format(str(round(executionTime_det, 2) / 60)))

executionTime_all = (time.time() - startTime_all)
print('Time taken for the entire webscrape: {} minutes.'.format(str(round(executionTime_all, 2) / 60)))
