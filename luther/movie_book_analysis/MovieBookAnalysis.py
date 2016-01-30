import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error



def getMergedDf(df1, df2, col1, col2='GRRating', col3='',col4='',col5='',col6='',col7=''):
    merged_data = pd.merge(df1, df2, on='Title')
    
    all_cols = [col1,col2,col3,col4,col5,col6,col7]
    col_list = ['Title']
    for col in all_cols:
        if col in merged_data.columns and col not in col_list:
            col_list.append(col)
    
    ratings_df = merged_data[col_list]
    try:
        ratings_df = ratings_df[ratings_df[col2] > 2.25]
        ratings_df = ratings_df.sort_values(by=col2, ascending=True)
        ratings_df = ratings_df[ratings_df[col2] > 0]
        ratings_df = ratings_df.dropna()
    except:
        pass

    
    return ratings_df, all_cols

def getTrainTestResultsBySeed(df1, df2, col1, col2='GRRating', col3='',col4='',col5='',col6='',col7='', transformDV='', transformIV1='', seed=42):
    merged_data, all_cols= getMergedDf(df1, df2, col1, col2, col3,col4,col5,col6,col7)
    merged_data = sm.add_constant(merged_data)
    cols = [col2,col3,col4,col4,col5,col6,col7, 'const']
    iv_cols = [col for col in cols if col != '']
    train, test = train_test_split(merged_data, test_size = 0.25, random_state=seed)
    
    train_results = getOLSResultsNoMerge(train, all_cols, col1, col2, col3,col4,col5,col6,col7, transformDV, transformIV1)
    test_results = getOLSResultsNoMerge(test, all_cols, col1, col2, col3,col4,col5,col6,col7, transformDV, transformIV1)
    return train, test, train_results, test_results

def trainTestEvaluation(df1, df2, col1, col2='GRRating', col3='',col4='',col5='',col6='',col7='', transformDV='', transformIV1=''):
    merged_data, all_cols= getMergedDf(df1, df2, col1, col2, col3,col4,col5,col6,col7)
    merged_data = sm.add_constant(merged_data)
    cols = [col2,col3,col4,col4,col5,col6,col7, 'const']
    iv_cols = [col for col in cols if col != '']
    iv_cols = list(set(iv_cols))
    print(iv_cols)
    train, test = train_test_split(merged_data, test_size = 0.25)
    
    train_results = getOLSResultsNoMerge(train, all_cols, col1, col2, col3,col4,col5,col6,col7, transformDV, transformIV1)
    print(train_results.summary())
    
    fig,ax = plt.subplots()
    fig.set_size_inches(10, 7, forward=True)
    test = test.sort_values(by=col2, ascending=True)
    plt.scatter(train[col2], train[col1], label='Training', color = 'red')
    plt.plot(test[col2], train_results.predict(test[iv_cols]), color='black',linewidth=1, label='Test')
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.legend(loc='upper left')
    plt.title('Training and Testing Data')
    
    

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

def formulaConstruction(ratings_df, all_cols, col1, col2, col3, col4, transformDV='', transformIV1=''):
    if transformDV == '':
        formula = col1 + ' ~ '
    elif transformDV == 'log':
        formula = 'np.log(' + col1 + ') ~ '
    elif transformDV == 'sqrt':
        formula = 'np.sqrt(' + col1 + ') ~ '
    elif transformDV == 'boxcox':
        formula = 'stats.boxcox(' + col1 + ')[0] ~ '
    else:
        return print('Bad transformDV variable')
    
    if col2 == '':
        formula += '1'
    
    if transformIV1 == '':
        formula += col2
    elif transformIV1 == 'square':
        formula += 'np.square(' + col2 + ')'
    elif transformIV1 == 'exp':
        formula += 'np.exp(' + col2 + ')'
    elif transformIV1 == 'log':
        formula += 'np.log(' + col2 + ')'
    else:
        return print('Bad transformIV1 variable')
    
    for colname in all_cols[2:]:
        if colname in ratings_df.columns:
            formula += (' + ' + colname)
    
    return formula
    
def getOLSResults(df1, df2, col1, col2='GRRating', col3='',col4='',col5='',col6='',col7='', transformDV='', transformIV1=''):
    ratings_df, all_cols = getMergedDf(df1, df2, col1, col2, col3, col4, col5, col6, col7)
    
    formula = formulaConstruction(ratings_df, all_cols, col1, col2, col3, col4, transformDV, transformIV1)
            
    print(formula)
    model = smf.ols(formula=formula, data=ratings_df)
    results = model.fit()
    return results

def getOLSResultsNoMerge(df, all_cols, col1, col2='GRRating', col3='',col4='',col5='',col6='',col7='', transformDV='', transformIV1=''):
    formula = formulaConstruction(df, all_cols, col1, col2, col3, col4, transformDV, transformIV1)
            
    print(formula)
    
    model = smf.ols(formula=formula, data=df)
    results = model.fit()
    return results
    

def plotByResults(results, title='', xlabel='', ylabel='', xlim=[2.5, 5], ylim=[7, 8.5], iv=1):
    fig,ax = plt.subplots()
    fig = sm.graphics.plot_fit(results, iv, ax=ax)
    fig.set_size_inches(10, 7, forward=True)
    fig.set_frameon(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)

def getQQPlot(results):
    measurements = results.resid
    fig,ax = plt.subplots()
    stats.probplot(measurements, dist="norm", plot=ax)
    fig.set_size_inches(10, 7, forward=True)


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