import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import os


def getMergedDf(df1, df2, col1, col2='GRRating', col3='',col4='',col5='',col6='',col7=''):
    merged_data = pd.merge(df1, df2, on='Title')
    
    all_cols = [col1,col2,col3,col4,col5,col6,col7]
    col_list = ['Title']
    for col in all_cols:
        if col in merged_data.columns:
            col_list.append(col)
    
    ratings_df = merged_data[col_list]
    ratings_df = ratings_df[ratings_df[col2] > 2.25]
    ratings_df = ratings_df.sort_values(by=col2, ascending=True)
    ratings_df = ratings_df[ratings_df[col2] > 0]
    ratings_df = ratings_df.dropna()
    
    return ratings_df, all_cols
    
def plotMovieRatingAgainstGR(df1, df2, col1, col2='GRRating', ymin=0, ymax=10, xmin=2, xmax=5.25, ylim=10):
    ratings_df = getMergedDf(df1, df2, col1, col2='GRRating')[0]
    
    
    ratings_df.plot(x=col2, y=col1, kind='scatter', figsize=(7,5))
    fit = np.polyfit(ratings_df[col2], ratings_df[col1], 1)
    line = np.poly1d(fit)
    xpoints = np.linspace(0.0,5.0,100)
    plt.plot(xpoints, line(xpoints), '-', color='b', label='Observed')
    
    factor = ymax//5
    factor2 = factor//2
    defaultx = [float(x)/factor for x in range(0,ymax+factor2)]
    defaulty = list(range(0,ymax+factor2))
    assert len(defaultx) == len(defaulty)
    plt.plot(defaultx,defaulty,color='r', label='Expected')

    plt.title(col1 + ' vs ' + col2)
    plt.legend(loc='upper left')
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.ylim([ymin,ylim])
    plt.xlim([xmin,xmax])
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    plt.savefig('figures/' + col1 + '_vs_' + col2 + '.png', transparent = False)

def plotByResults(results, title='', xlabel='', ylabel='', xlim=[2.5, 5], ylim=[7, 8.5]):
    fig,ax = plt.subplots()
    fig = sm.graphics.plot_fit(results, 1, ax=ax)
    fig.set_size_inches(10, 7, forward=True)
    fig.set_frameon(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

def getQQPlot(results):
    measurements = results.resid
    stats.probplot(measurements, dist="norm", plot=plt)
    plt.show()
    
    
def getOLSResults(df1, df2, col1, col2='GRRating', col3='',col4='',col5='',col6='',col7='', transformDV='', transformIV1=''):
    ratings_df, all_cols = getMergedDf(df1, df2, col1, col2, col3, col4, col5, col6, col7)
    
    if transformDV == '':
        formula = col1 + ' ~ '
    elif transformDV == 'log':
        formula = 'np.log(' + col1 + ') ~ '
    elif transformDV == 'sqrt':
        formula = 'np.sqrt(' + col1 + ') ~ '
    elif transformDV == 'boxcox':
        formula = 'stats.boxcox(' + col1 + ')[0] ~ '
        
    if transformIV1 == '':
        formula += col2
    elif transformIV1 == 'square':
        formula += 'np.square(' + col2 + ')'
    elif transformIV1 == 'exp':
        formula += 'np.exp(' + col2 + ')'
    elif transformIV1 == 'log':
        formula += 'np.log(' + col2 + ')'
    else:
        return print('Bad TransformIV1 variable')
    
    for colname in all_cols[2:]:
        if colname in ratings_df.columns:
            formula += (' + ' + colname)
            
    print(formula)
    
    model = smf.ols(formula=formula, data=ratings_df)
    results = model.fit()
    return results

def getZscoreOfCol(df,col):
    df[(col+'_zscore')] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    
    return df

def getZscoreDiffofMovieAndBook(moviedf, GRdf, moviecol, GRcol='GRRating', want='bettermovies'):
    assert want == 'bettermovies' or want == 'betterbooks'
    movie_z = moviecol+'_zscore'
    GR_z = GRcol+'_zscore'
    
    moviedf = moviedf[moviedf[moviecol] > 0]
    GRdf = GRdf[GRdf[GRcol] > 0]
    z_movie_rating = getZscoreOfCol(moviedf, moviecol)
    z_GR_ratings = getZscoreOfCol(GRdf, GRcol)
    merged_df = pd.merge(z_movie_rating, z_GR_ratings, on='Title')
    if want == 'bettermovies':
        merged_df['ZS_diff'] = merged_df[movie_z] - merged_df[GR_z]
    elif want == 'betterbooks':
        merged_df['ZS_diff'] = merged_df[GR_z] - merged_df[movie_z]
    
    zs_diff_df = merged_df[['Title', moviecol, GRcol, 'ZS_diff']]
    sort_zs = zs_diff_df.sort_values(by='ZS_diff', ascending=False)
    
    return sort_zs