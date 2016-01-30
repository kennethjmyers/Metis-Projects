import requests
from bs4 import BeautifulSoup
import re
import dateutil.parser
import pandas as pd
from collections import defaultdict


def readYearPage(year_page_url):
    response = requests.get(year_page_url)
    page = response.text
    soupPage = BeautifulSoup(page, 'lxml')
  
    return soupPage
   
def getMoviesAndLinks(year_page_url):
    movie_page = readYearPage(year_page_url)
    
    movie_name_list = []
    movie_link_list = []

    movie_data = movie_page.find_all('a', href=re.compile("/movies/\?id="))
    del movie_data[0]

    for movie in movie_data:
        movie_name_list.append(movie.text)
        movie_link_list.append('http://www.boxofficemojo.com' + movie['href'])
        
    return list(zip(movie_name_list, movie_link_list))

def get_movie_value(soup, field_name):
    obj = soup.find(text=re.compile(field_name))
    if not obj: 
        return None
    
    next_sibling = obj.findNextSibling()
    next_value = obj.find_next()  #sometimes findNextSibling doesnt work if string isnt in quotes
    
    if next_sibling:
        return next_sibling.text
    elif next_value:
        return next_value.text
    else:
        return None


def to_date(datestring):
    date = dateutil.parser.parse(datestring)
    return date

def budget_to_int(moneystring):
    try:
        moneystring = moneystring.replace('$', '').replace(' million', '000000')
        return int(moneystring)
    except:
        return ''

def money_to_int(moneystring):
    try:
        moneystring = moneystring.replace('$', '').replace(',', '')
        return int(moneystring)
    except: 
        return ''

def runtime_to_minutes(runtimestring):
    runtime = runtimestring.split()
    try:
        minutes = int(runtime[0])*60 + int(runtime[2])
        return minutes
    except:
        return ''
    
def scrapeOneMoviePage(movie_page_url):
    response = requests.get(movie_page_url)
    page = response.text
    soupPage = BeautifulSoup(page, 'lxml')
    
    raw_budget = get_movie_value(soupPage, 'Production Budget')
    budget = budget_to_int(raw_budget)

    raw_dtg = get_movie_value(soupPage, 'Domestic Total')
    dtg = money_to_int(raw_dtg)

    try:
        director = soupPage.find('a', href=re.compile('Director')).find_next().text
    except:
        director = ''
        
    rating = get_movie_value(soupPage, 'MPAA Rating:')
    
    raw_runtime = get_movie_value(soupPage, 'Runtime')
    runtime = runtime_to_minutes(raw_runtime)
    
    raw_releasedate = get_movie_value(soupPage, 'Release Date:')
    releasedate = to_date(raw_releasedate)
    
    movie_data = [budget, dtg, director, rating, runtime, releasedate]
    
    return movie_data

def scrapeAllMoviesPages(year_page_url):
    years_movie_names_and_links = getMoviesAndLinks(year_page_url)
    
    
    movie_link_list = []

    movie_title_list = []
    movie_budget_list = []
    movie_dtg_list = []
    movie_director_list = []
    movie_rating_list = []
    movie_runtime_list = []
    movie_releasedate_list = []

    movie_data_list = [movie_title_list, movie_budget_list, movie_dtg_list, 
                      movie_director_list, movie_rating_list, movie_runtime_list, 
                      movie_releasedate_list]

    for movie in years_movie_names_and_links:
        movie_data_list[0].append(movie[0])
        movie_link_list.append(movie[1])

    for link in movie_link_list:
        page_data = scrapeOneMoviePage(link)
        for i in range(1, len(page_data)+1):
            movie_data_list[i].append(page_data[i-1])
            
    return movie_data_list

def generateDataFrame(year_page_url):
    movie_data = scrapeAllMoviesPages(year_page_url)
    header = ['Title', 'Budget', 'DomesticTotalGross', 'Director', 'Rating', 'Runtime', 'ReleaseDate']
    movie_data_dict = defaultdict(list)

    for _ in range(len(header)):
        movie_data_dict[header[_]] = movie_data[_]
        
    movie_df = pd.DataFrame(movie_data_dict)
    movie_df = movie_df[header]
    
    return movie_df

def write_to_csv(year_page_url, outfilename):
    movie_df = generateDataFrame(year_page_url)
    
    #print(movie_df.head())
    
    movie_df.to_csv(outfilename)