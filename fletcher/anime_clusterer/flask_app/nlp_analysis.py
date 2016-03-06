import numpy as np
from numpy import zeros
import pandas as pd
import pickle

from random import sample
from random import random
from collections import defaultdict

from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.decomposition import RandomizedPCA

import nltk
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


import re
import os
import codecs
from sklearn import feature_extraction
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpld3

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text.replace("'", '')) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    #stems = [stemmer.stem(t) for t in filtered_tokens]
    stems = [lemmatizer.lemmatize(t, pos='v') for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text.replace("'", '')) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def clean_synops(synopses, nouns_only = False):
    cleaned_synopses = []
    for synopsis in synopses:
        cleaned_tokens = tokenize_and_stem(synopsis)
        cleaned_tokens = [word.lower() for word in cleaned_tokens]
        if nouns_only:
            tags = tagger.tag(cleaned_tokens)
            cleaned_tokens = [t[0] for t in tags if t[1] == "NN"]

        cleaned_synopses.append(" ".join(cleaned_tokens))
        
    return cleaned_synopses

class cluster_analysis(object):
    
    #Perform the preprocessing#
    def __init__(self):
        #import colors
        with open('pickles/tableaucolors.pkl', 'rb') as file:
            self.tbl_colors = pickle.load(file)
    
        #this is for speeding up pos_tag() since pos_tag() is very slow due to unpickling each time
        self.tagger = PerceptronTagger() 
        
        #import data
        self.full_data = pd.read_csv('full_anime_data_set.csv')
        self.full_data.dropna(subset=['synopsis'], inplace=True)
        self.full_data['titlelower'] = self.full_data['title'].str.lower()
        self.synops = list(self.full_data['synopsis'])
        
        #remove short synopses
        short_indices = []
        for i, synop in enumerate(self.synops):
            if len(synop.split()) <= 25:
                short_indices.append(i)

        synops = [synop for i, synop in enumerate(self.synops) if i not in short_indices]
        self.full_data = self.full_data.drop(self.full_data.index[short_indices])
        
        #create vocab frame for easier indexing of stemmed words
        self.totalvocab_stemmed = []
        self.totalvocab_tokenized = []
        for i in self.synops:
            self.allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
            self.totalvocab_stemmed.extend(self.allwords_stemmed) #extend the 'totalvocab_stemmed' list

            self.allwords_tokenized = tokenize_only(i)
            self.totalvocab_tokenized.extend(self.allwords_tokenized)

        self.vocab_frame = pd.DataFrame({'words': self.totalvocab_tokenized}, index = self.totalvocab_stemmed)
        
        #Some of the above is necessary, some is not I included it all incase it was necessary one day
        
        #clean sentences
        self.cleaned_synops = clean_synops(self.synops)
        
        #unpickle the tfidf_matrix, vectorizer and the Kmeans object
        self.num_clusters = 9
        self.tfidf_vectorizer = joblib.load('pickles/tfidf_vectorizer.pkl')
        self.tfidf_matrix = joblib.load('pickles/tfidf_matrix.pkl')
        self.km = joblib.load('pickles/km.pkl')
        
        self.terms = self.tfidf_vectorizer.get_feature_names()
        
        #Change dataframe to have clusters as indices
        self.clusters = list(self.km.labels_)
        self.full_data['clusters'] = self.clusters
        self.full_data.set_index('clusters', inplace=True)
        
        #Create cluster 
        self.cluster_names = defaultdict(str)
        order_centroids = self.km.cluster_centers_.argsort()[:, ::-1] 

        for i in range(self.num_clusters):
            #print("Cluster %d words:" % i, end='')

            temp_list = []
            for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
                try:
                    temp_list.append(self.terms[ind])
                except:
                    pass
            #the line below is for creating labels for the graph using the first threw words
            self.cluster_names[i] = ', '.join(temp_list[:5])
            
        print("All clear")
        
    #Plot the model
    def getPlot(self, series_title, rec_num, cluster_option):
        title_list_lower = list(self.full_data.titlelower)
        title_list = list(self.full_data.title)
        #get index of desired title
        titles_index = title_list_lower.index(series_title.lower())
        
        #get indices where clusters are same as that of desired title
        good_indices = [i for i in range(len(self.clusters)) if self.clusters[i] == self.clusters[titles_index]]
        
        if cluster_option:
            temp_tfidf_matrix = [self.tfidf_matrix[i] for i in good_indices]
            temp_clusters = [self.clusters[i] for i in good_indices]
            temp_full_data = self.full_data.iloc[good_indices]
            title_list = [title_list[i] for i in good_indices]
        else:
            temp_tfidf_matrix = self.tfidf_matrix
            temp_clusters = self.clusters
            temp_full_data = self.full_data
        
        
        #get pwd between each point and desired title
        pwd = pairwise_distances(temp_tfidf_matrix,\
                                 self.tfidf_matrix[titles_index].reshape(1,-1),\
                                 metric='cosine')
        
        #reshape pwd
        pwd = [i[0] for i in pwd]
        #get the 10 best + original title
        closest = np.array(pwd).argsort()[:int(rec_num)+1]
        closest_loc = [temp_tfidf_matrix[i] for i in closest]
        #calc distance
        dist = 1 - cosine_similarity(closest_loc)
        #used for labeling
        subset_clusters = [temp_clusters[i] for i in closest]
        #recommendation list titles
        titles = [title_list[i] for i in closest]
        
        #compute multidimensional scaling of the titles into 2 dimensions
        mds = MDS(n_components=2, dissimilarity="precomputed", n_init=3, max_iter=100, n_jobs=-2, random_state=1)
        pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

        xs, ys = pos[:, 0], pos[:, 1]
        
        #create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x=xs, y=ys, label=subset_clusters, title=titles)) 

        #group by cluster
        groups = df.groupby('label')

        #define custom css to format the font and to remove the axis labeling
        css = """
        text.mpld3-text, div.mpld3-tooltip {
          font-family:Arial, Helvetica, sans-serif;
        }

        g.mpld3-xaxis, g.mpld3-yaxis {
        display: none; }
        
        svg.mpld3-figure {
        transform: 0;
        display: block;
        margin: 0 auto;
        
        /*
        margin-left: -200px;
        */
        }
        
        """

        # Plot 
        fig, ax = plt.subplots(figsize=(15,7.65)) #set plot size
        ax.margins(0.03)# Optional, just adds 5% padding to the autoscaling
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])

        #iterate through groups to layer the plot
        #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        i=0
        for name, group in groups:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=15, 
                             label=self.cluster_names[name], mec='none', 
                             color=self.tbl_colors[i]
                            )
            ax.set_aspect('equal')
            labels = [i for i in group.title]
            ax.legend(loc='best')

            #set tooltip using points, labels and the already defined 'css'
            tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                               voffset=10, hoffset=10, css=css)
            #connect tooltip to fig
            mpld3.plugins.connect(fig, tooltip, TopToolbar())    

            #set tick marks as blank
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            #set axis as blank
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            i+=1
            
        #add permanent label for searched series
        ax.text(df.ix[0]['x'], df.ix[0]['y'], df.ix[0]['title'], size=16)


        ax.legend(loc='best', title='', fancybox=True, numpoints=1) #show legend with only one dot

        #print(mpld3.fig_to_dict(fig))
        html = mpld3.fig_to_html(fig)
        
        subset_data = temp_full_data.iloc[closest]
        subset_data.reset_index(inplace=True)
        #subset_data.set_index('title', drop=False, inplace=True)
        subset_data = subset_data.to_json(orient='records')
        
        return html, subset_data

        

#Creates a toolbar for mpld3 plots
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 600);
      this.fig.toolbar.toolbar.attr("y", 0);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}
        
    
    
    
    
    
    
    