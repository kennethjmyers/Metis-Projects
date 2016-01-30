import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from collections import defaultdict
import html5lib


def readPage(year_page_url):
    response = requests.get(year_page_url)
    page = response.text
    soupPage = BeautifulSoup(page, 'lxml')
  
    return soupPage
 
def scrapeSearchPage(page_url):
    page_data = readPage(page_url)
    
    movie_name_list = []
    movie_link_list = []
    
    movie_data = page_data.find_all('a', href=re.compile('^/title'), title='')
    #next_link = 'http://www.imdb.com' + page_data.find(text=re.compile('^Next')).parent['href']
    
    for movie in movie_data:
        movie_name_list.append(movie.text)
        movie_link_list.append('http://www.imdb.com' + movie['href'])
        
    
    return list(zip(movie_name_list[1:], movie_link_list[1:]))
    

def scrapeMultiPages(first_page_url, num_pages=4):
    movie_list = []
    next_page_url = first_page_url
    current_page_num = 1
    page_equals = 'page='
    
    for _ in range(num_pages):
        movies = scrapeSearchPage(next_page_url)
        next_page_url = next_page_url.replace(page_equals+str(current_page_num), page_equals+str(current_page_num+1))
        for movie in movies:
            if movie[0] == ' \n' or 'See full' in movie[0]:
                pass
            else:
                movie_list.append(movie)
            
    return movie_list

def getMovieValue(soup, itemprop_value):
    if itemprop_value == 'duration':
        value = soup.find('time', itemprop=itemprop_value)['datetime']
        if len(value) == 6:
            return value[2:5]
        elif len(value) == 5:
            return value[2:4]
    elif itemprop_value == 'contentRating':
        value = soup.find('meta', itemprop=itemprop_value)['content']
        return value
    else:
        value = soup.find('span', itemprop=itemprop_value)
        return value.text

def scrapeMoviePage(page_link):
    page = readPage(page_link)
    
    #RatingValue
    try:
        ratingValue = float(getMovieValue(page, 'ratingValue'))
    except AttributeError:
        ratingValue = None
    
    #RatingCount
    try:
        ratingCount = int(getMovieValue(page, 'ratingCount').replace(',', ''))
    except AttributeError:
        ratingCount = None
        
    #Runtime
    try:
        runtime = int(getMovieValue(page, 'duration').replace(' min', ''))
    except:
        runtime = None
    #MPAARating
    try:
        MPAARating = getMovieValue(page, 'contentRating')
    except:
        MPAARating = None
    
    #Budget
    try:
        subheadings = page.find_all('h4', class_='inline')
        subheadings_text = [i.text for i in subheadings]
        #print(subheadings)
        if 'Budget:' in subheadings_text:
            for i in subheadings:
                if i.text == 'Budget:':
                    budget = int(i.parent.text.split()[1].replace('$', '').replace(',',''))
        else: 
            budget = None
    except:
        budget = None
    
    #Gross
    try:
        subheadings = page.find_all('h4', class_='inline')
        subheadings_text = [i.text for i in subheadings]
        if 'Gross:' in subheadings_text:
            for i in subheadings:
                if i.text == 'Gross:':
                     gross = int(i.parent.text.split()[1].replace('$', '').replace(',',''))
        else: 
            gross = None
    except:
        gross = None 
    
    return [ratingValue, ratingCount, runtime, MPAARating, budget, gross]

def scrapeAllMoviesPages(year_page_url, num_pages=4):
    movie_names_links = scrapeMultiPages(year_page_url, num_pages)
    
    movie_names = []
    movie_ratings = []
    movie_ratings_count = []
    movie_runtimes = []
    content_ratings = []
    movie_budgets = []
    movie_grosses = []

    
    for movie in movie_names_links:
        rating_and_count = scrapeMoviePage(movie[1])
        movie_names.append(movie[0])
        movie_ratings.append(rating_and_count[0])
        movie_ratings_count.append(rating_and_count[1])
        movie_runtimes.append(rating_and_count[2])
        content_ratings.append(rating_and_count[3])
        movie_budgets.append(rating_and_count[4])
        movie_grosses.append(rating_and_count[5])
        
    
    return [movie_names, movie_budgets, movie_grosses, content_ratings, movie_runtimes, movie_ratings, movie_ratings_count]

def generateIMDBDataFrame(year_page_url, num_pages=4):
    movie_data = scrapeAllMoviesPages(year_page_url, num_pages)
    header = ['Title', 'Budget', 'Gross', 'MPAArating', 'Runtime', 'IMDBRating', 'IMDBRatingCount']
    movie_data_dict = defaultdict(list)

    for _ in range(len(header)):
        movie_data_dict[header[_]] = movie_data[_]
        
    movie_df = pd.DataFrame(movie_data_dict)
    movie_df = movie_df[header]
    
    return movie_df

def writeIMDBToCSV(year_page_url, outfilename, num_pages=4):
    movie_df = generateIMDBDataFrame(year_page_url, num_pages)
    
    #print(movie_df.head())
    
    movie_df.to_csv(outfilename)