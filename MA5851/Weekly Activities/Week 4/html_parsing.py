# Step 1: view the website
# Step 2: download the website
import requests

page = requests.get("https://www.smh.com.au/")
print(page)

print(page.status_code) # if this = 200 then the website was successfully downloaded (smth starting with 4 or 5 generally
                        #  indicates an error)
#print(page.content) # prints the page content

# Step 3: parsing a page with BeautifulSoup
from bs4 import BeautifulSoup

soup = BeautifulSoup(page.content, 'html.parser')
#print(soup.prettify()) # prints the page content nicely formatted

children = list(soup.children) # select all the elements at the top level of the page. children calls a list generator so
                                #   need to call a list function on it
#print(children)

print([type(item) for item in list(soup.children)])

html = list(soup.children)[1]
#print(html)
#list(html.children)

head = list(html.children)[0]
head_items = list(head.children)
print(len(head_items))

title_text = head_items[23].attrs['content']
title_text