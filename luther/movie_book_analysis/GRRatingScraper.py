import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
from collections import defaultdict
import pandas as pd

key = ''
search_key = 'https://www.goodreads.com/search/index.xml?key=' + key + '&q='

movies_based_on_books = 'http://www.imdb.com/search/keyword?keywords= \
                         based-on-novel&mode=advanced&page=1&title_type= \
                         movie&ref_=kw_vw_adv&sort=user_rating,desc'

def readPage(search_page_url):
    response = requests.get(search_page_url)
    page = response.text
    soup = BeautifulSoup(page, 'xml')
  
    return soup

def getBookData(movie_book_list, key=key, search_key=search_key):
    movie_book_names = []
    movie_book_ratings = []
    movie_book_ratings_counts = []
    
    got_books, missed_books = 0, 0

    for movie in movie_book_list:
        page_data = readPage(search_key + urllib.parse.quote(movie))
        total_results = page_data.find('total-results').text
        if int(total_results) > 1:
            rating = float(page_data.find('average_rating').text)
            rating_count = int(page_data.find('ratings_count').text)
            
            movie_book_names.append(movie)
            movie_book_ratings.append(rating)
            movie_book_ratings_counts.append(rating_count)
            got_books += 1
            #print('Got {} books'.format(str(got_books)))
        else:
            missed_books += 1
            #print('Missed {} books'.format(str(missed_books)))
        
        time.sleep(1)
            
    return [movie_book_names, movie_book_ratings, movie_book_ratings_counts]
        
def generateGRDataFrame(movie_book_data):
    header = ['Title', 'GRRating', 'GRRatingCount']
    movie_book_data_dict = defaultdict(list)

    for _ in range(len(header)):
        movie_book_data_dict[header[_]] = movie_book_data[_]
        
    movie_book_df = pd.DataFrame(movie_book_data_dict)
    movie_book_df = movie_book_df[header]
    
    return movie_book_df

def writeGRToCSV(movie_book_data, outfilename):
    movie_book_df = generateGRDataFrame(movie_book_data)
    
    #print(movie_df.head())
    
    movie_book_df.to_csv(outfilename)