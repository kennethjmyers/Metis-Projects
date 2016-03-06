# Anime Recommender

## What is this?

This was my fourth project at Metis, Project Fletcher. The goal was to apply NLP analysis techniques to a topic of choice. I decided to make an Anime Recommender using NLP, tf-idf, and K-means.

Here are some helpful links about this project:

* [My blog](http://kennmyers.github.io/kennmyers.github.io/data%20science/Metis-Fourth-Project) where I describe my process.
* [The final product](https://anime-recommender.herokuapp.com/). Please note it can take up to ~15 seconds to load, I am hosting it on a free Heroku account.

## The files

Below is a detailed list of the files in this directory:

* **flask_app/** : This is where the files needed to run the flask app are located. I believe they are complete and can be run locally on your machine by calling ```python flask_app.py``` and then opening 0.0.0.0:8000 in the browser.
* **prescraped_list.csv** : This is the file I started with. I found it on Reddit and used it to save time on scraping the list of anime on MyAnimeList. Because of this, the list of anime does not go past the end of 2014.
* **get_show_ids.ipynb** : This file takes the prescraped_list.csv and converts it into a more useable format. Results are outputted to ```shows_with_ids.csv```.
* **shows_with_ids.csv** : This is the file outputted by ```get_show_ids.ipynb```.
* **mal_scraping.ipynb** : Scrapes MyAnimeList using the names from ```shows_with_ids.csv``` and merges that file with the new data. Outputs to ```full_anime_data_set.csv```
* **mal_config.txt** : required information to access the MyAnimeList API which is used in ```mal_scraping.ipynb```. No spaces.
* **full_anime_data_set.csv** : main data file containing all information. Used in the flask app.
* **export_titles.ipynb** : Takes the ```full_anime_data_set.csv``` and outputs a list of just the anime titles into the flask app. This is used for the autocomplete feature in the apps search bar.
* **metis_challenge_13.ipynb** : This was where I did some preliminary work while deciding how to procede with the nlp analysis of the anime descriptions. 
