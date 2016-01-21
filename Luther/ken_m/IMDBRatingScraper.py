import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from collections import defaultdict


def readPage(year_page_url):
    response = requests.get(year_page_url)
    page = response.text
    soupPage = BeautifulSoup(page, 'lxml')
  
    return soupPage
 
def scrapeYearPage(page_url):
    page_data = readPage(page_url)
    
    movie_name_list = []
    movie_link_list = []
    
    movie_data = page_data.find_all('a', href=re.compile('^/title'), title='')
    next_link = 'http://www.imdb.com' + page_data.find(text=re.compile('Next')).parent['href']
    
    for movie in movie_data:
        movie_name_list.append(movie.text)
        movie_link_list.append('http://www.imdb.com' + movie['href'])
        
    
    return list(zip(movie_name_list, movie_link_list)), next_link

def scrapeYearMultiPages(first_page_url, num_pages=2):
    movie_list = []
    next_page_url = first_page_url
    
    for _ in range(num_pages):
        movies, next_page_url = scrapeYearPage(next_page_url)
        for movie in movies:
            movie_list.append(movie)
            
    return movie_list

def getMovieValue(soup, itemprop_value):
    value = soup.find('span', itemprop=itemprop_value)
    return value.text

def scrapeMoviePage(page_link):
    page = readPage(page_link)
    
    ratingValue = float(getMovieValue(page, 'ratingValue'))
    ratingCount = int(getMovieValue(page, 'ratingCount').replace(',', ''))
    
    return [ratingValue, ratingCount]

def scrapeAllMoviesPages(year_page_url):
    movie_names_links = scrapeYearMultiPages(year_page_url)
    
    movie_names = []
    movie_ratings = []
    movie_ratings_count = []
    
    for movie in movie_names_links:
        rating_and_count = scrapeMoviePage(movie[1])
        movie_names.append(movie[0])
        movie_ratings.append(rating_and_count[0])
        movie_ratings_count.append(rating_and_count[1])
    
    return [movie_names, movie_ratings, movie_ratings_count]

def generateIMDBDataFrame(year_page_url):
    movie_data = scrapeAllMoviesPages(year_page_url)
    header = ['Title', 'Rating', 'RatingCount']
    movie_data_dict = defaultdict(list)

    for _ in range(len(header)):
        movie_data_dict[header[_]] = movie_data[_]
        
    movie_df = pd.DataFrame(movie_data_dict)
    movie_df = movie_df[header]
    
    return movie_df

def writeIMDBToCSV(year_page_url, outfilename):
    movie_df = generateIMDBDataFrame(year_page_url)
    
    #print(movie_df.head())
    
    movie_df.to_csv(outfilename)