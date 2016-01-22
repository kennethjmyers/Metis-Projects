import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from collections import defaultdict
import IMDBRatingScraper
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
import time
import os


def getMoviesList(imdb_yearly_url):
    movies_list = []
    movies_info = IMDBRatingScraper.scrapeYearMultiPages(imdb_yearly_url)
    for movie in movies_info:
        movies_list.append(movie[0])
        
    return movies_list

def scrapeMetaCritic(movies_list):
    #print(movies_list)
    time.sleep(2)
    
    movie_names = []
    movie_ratings = []
    movie_ratings_count = []
    critic_ratings = []
    critic_ratings_count = []
    
    driver_options = webdriver.ChromeOptions()
    driver_options.add_extension('Adblock-Plus_v1.10.crx')
    
    chromedriver = '/Users/kenn/Downloads/KEEP/chromedriver'
    os.environ['webdriver.chrome.driver'] = chromedriver
    driver = webdriver.Chrome(chromedriver, chrome_options = driver_options)
    driver.get('http://www.metacritic.com/')
    time.sleep(2)
    
    for movie in movies_list:
        movie_name_holder = movie
        if '- ' in movie:
            movie = movie.replace('- ', '')
        
        #starting page
        #word = "Star wars the force awakens"
        page_is_good = False
        while not page_is_good:
            try:
                WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, 'masthead_search_term')))
                #print("Page is ready!")
                page_is_good = True
            except TimeoutException:
                print("Loading page took too much time!")
                driver.refresh()
        search_form = driver.find_element_by_id('masthead_search_term')
        search_form.send_keys(movie)
        search_form.send_keys(Keys.RETURN)

        #search page
        page_is_good = False
        result_check = driver.find_element_by_tag_name('p').text
        if 'No search results found.' in result_check:
            search_form = driver.find_element_by_id('masthead_search_term')
            search_form.clear()
            continue
        while not page_is_good:
            try:
                WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.XPATH, "//li[@class='result first_result']")))
                #print("Page is ready!")
                page_is_good = True
            except TimeoutException:
                print("Loading page took too much time!")
                driver.refresh()
        try:
            i = 1
            result_type_text = ''
            while result_type_text != 'Movie':
                result_type = driver.find_element_by_xpath("(//div[@class='result_type'])[" + str(i) + "]")
                result_type_text = result_type.find_element_by_tag_name('strong').text
                #print(result_type_text)
                #print(i)
                i += 1    

            result_type_parent = result_type.find_element_by_xpath('..')
            #print(result_type_parent.text)
            #result = result_type_parent.find_element_by_xpath(".//li[contains(@class, 'result')]")
            result_link = result_type_parent.find_element_by_xpath(".//a")
            result_link_text = result_link.text
            if movie[:int(len(movie)/2)] not in result_link_text:
                search_form = driver.find_element_by_id('masthead_search_term')
                search_form.clear()
                continue
            else:
                result_link.click()
        except NoSuchElementException:
            search_form = driver.find_element_by_id('masthead_search_term')
            search_form.clear()
            continue

        #movie page
        page_is_good = False
        try:
            unreleased_check = driver.find_element_by_class_name('countdown_msg').text
            if 'until movie release' in unreleased_check:
                search_form = driver.find_element_by_id('masthead_search_term')
                search_form.clear()
                continue
        except NoSuchElementException:
            pass
            
        while not page_is_good:
            try:
                #WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'metascore_w user')]")))
                #print("Page is ready!")
                page_is_good = True
            except TimeoutException:
                print("Loading page took too much time!")
                driver.refresh()
        try:
            user_rating = float(driver.find_element_by_xpath("//div[contains(@class, 'metascore_w user')]").text)
        except ValueError:
            user_rating = None
        except NoSuchElementException:
            search_form = driver.find_element_by_id('masthead_search_term')
            search_form.clear()
            continue
        
        try:
            user_rating_count = driver.find_element_by_xpath("(//span[@class='count'])[2]").find_element_by_tag_name('a').text
            user_rating_count = int(user_rating_count.split()[0])
        except:
            user_rating_count = None
        
        try:
            critic_rating = float(driver.find_element_by_xpath("//span[@itemprop='ratingValue']").text)
        except:
            critic_rating = None
            
        try:    
            critic_rating_count = driver.find_element_by_xpath("(//span[@class='count'])[1]").find_element_by_tag_name('a').text
            critic_rating_count = int(critic_rating_count.split()[0])
        except:
            critic_rating_count = None
        
        movie_names.append(movie_name_holder)
        movie_ratings.append(user_rating)
        movie_ratings_count.append(user_rating_count)
        critic_ratings.append(critic_rating)
        critic_ratings_count.append(critic_rating_count)

    driver.close()
    #print(movie_names, movie_ratings, movie_ratings_count, critic_ratings, critic_ratings_count)
    
    return[movie_names, movie_ratings, movie_ratings_count, critic_ratings, critic_ratings_count]


def generateMCDataFrame(movie_data):
    #movie_data = scrapeAllMoviesPages(year_page_url)
    header = ['Title', 'MCUserRating', 'MCUserRatingCount', 'MCCriticRating', 'MCCriticRatingCount']
    movie_data_dict = defaultdict(list)

    for _ in range(len(header)):
        movie_data_dict[header[_]] = movie_data[_]
        
    movie_df = pd.DataFrame(movie_data_dict)
    movie_df = movie_df[header]
    
    return movie_df

def writeMCToCSV(movie_df, outfilename):
    #movie_df = generateIMDBDataFrame(year_page_url)
    
    #print(movie_df.head())
    
    movie_df.to_csv(outfilename)
    