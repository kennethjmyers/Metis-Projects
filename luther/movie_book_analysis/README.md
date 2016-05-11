#READ ME

This repo contains my files from the second Metis project. The code is written in the *.py files and executed in the *.ipynb files.

###Python files

[MojoScraper.py](https://github.com/kennmyers/Metis-Projects/blob/master/luther/movie_book_analysis/MojoScraper.py) scrapes Box Office Mojo using BS4

[IMDBRatingScraper.py](https://github.com/kennmyers/Metis-Projects/blob/master/luther/movie_book_analysis/IMDBRatingScraper.py) scrapes IMDB using BS4

[GRRatingScraper.py](https://github.com/kennmyers/Metis-Projects/blob/master/luther/movie_book_analysis/GRRatingScraper.py) scrapes GoodReads using their API and BS4

[MCRatingScraper.py](https://github.com/kennmyers/Metis-Projects/blob/master/luther/movie_book_analysis/MCRatingScraper.py) scrapes MetaCritic using the Selenium

[MovieBookAnalysis.py](https://github.com/kennmyers/Metis-Projects/blob/master/luther/movie_book_analysis/MovieBookAnalysis.py) Uses various plotting and analysis libraries to analyze the data

[CrossValidation.py](https://github.com/kennmyers/Metis-Projects/blob/master/luther/movie_book_analysis/CrossValidation.py)  Uses mostly Sci Kit Learn to cross validate different regression models of the data

###Changes you will need to make to use these files

In order to use the GoodReads scraper you will need to register as a developer with [GoodReads and obtain a key](https://www.goodreads.com/api). This should be placed into its respective variable in the GRRatingScraper.py file.

If you do not wish to use my method of scraping Metacritic you could always use their API. Otherwise follow the instructions below.

In order to scrape MetaCritic using my method, you will need to download the [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/) and save it to a a directory named '~/Downloads/KEEP' or specify a different location in the MCRatingScraper.py file. 

You will also need to download the [AdBlockPlus extension](https://chrome.google.com/webstore/detail/adblock-plus/cfhdojbkjhnklbpkdaibdccddilifddb?hl=en-US) (you can use the [Chrome Extension Downloader](http://chrome-extension-downloader.com/)) or a similar Ad Blocker to scrape MetaCritic with selenium. The ad block extension should be stored in this directory and specified in MCRatingScraper.py. 


##Additional Information

Here is a [link to my blog post](http://kennmyers.github.io/kennmyers.github.io/data%20science/Metis-Second-Project/) about my findings.


Link to the [presentation](https://docs.google.com/presentation/d/1LYB7e1kdGuIurbkghLURA3gfZv6MLUNcuCLfptaj5U0/edit?usp=sharing)
