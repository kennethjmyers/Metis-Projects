from MovieBookAnalysis import *
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from copy import deepcopy


def avgMSEForLinReg(df, IV, DVs, k=10):
    MSE_list = []
    
    var_list = DVs
    var_list.insert(0, IV)
    tempDF = df[var_list].dropna()
    
    for x in range(k):
        y_train, y_test, X_train, X_test = train_test_split(tempDF[IV], tempDF[DVs], test_size=0.2)

        model = LinearRegression(normalize=True)
        model.fit(X_train, y_train)

        MSE = np.mean((model.predict(X_test)-y_test)**2)
        MSE_list.append(MSE)
        
    return np.mean(MSE_list)

def CrossValidateForMSEAtDiffLambda(df, IV, DVs):
    
    avgMSELasso_values = []
    avgMSERidge_values = []
    avgMSELinReg_values = []
    avgR2Lasso_values = []
    avgR2Ridge_values = []
    avgR2LinReg_values = []
    
    
    lmda_values = [x*0.00005 for x in range(1, 1001)]
    #lmda_values = [x*0.00005 for x in range(1, 11)]
    
    var_list = deepcopy(DVs)
    var_list.insert(0, IV)
    tempDF = df[var_list].dropna()
    
    for lmda in lmda_values:
        MSELasso_list = []
        MSERidge_list = []
        MSELinReg_list = []
        R2Lasso_list = []
        R2Ridge_list = []
        R2LinReg_list = []
        
        for x in range(10):
            y_train, y_test, X_train, X_test  = train_test_split(tempDF[IV], tempDF[DVs], test_size=0.2)
            
            #print(X_train.shape,X_test.shape, y_train.shape, y_test.shape)
            #print(train_test_split(tempDF[IV], tempDF[DVs], test_size=0.2))
            
            model = Lasso(alpha=lmda, normalize=True)
            model2 = Ridge(alpha=lmda, normalize=True)
            model3 = LinearRegression(normalize=True)
            model.fit(X_train, y_train)
            model2.fit(X_train, y_train)
            model3.fit(X_train, y_train)
            #print(model.score(X_test, y_test), model2.score(X_test, y_test))
            MSELasso = np.mean((model.predict(X_test)-y_test)**2)
            MSELasso_list.append(MSELasso)
            MSERidge = np.mean((model2.predict(X_test)-y_test)**2)
            MSERidge_list.append(MSERidge)
            MSELinReg = np.mean((model3.predict(X_test)-y_test)**2)
            MSELinReg_list.append(MSELinReg)
            R2Lasso_list.append(model.score(X_test, y_test))
            R2Ridge_list.append(model2.score(X_test, y_test))
            R2LinReg_list.append(model3.score(X_test, y_test))
            #print(X_test, y_test)
            
            
        avgMSELasso_values.append(np.mean(MSELasso_list))
        avgMSERidge_values.append(np.mean(MSERidge_list))
        avgMSELinReg_values.append(np.mean(MSELinReg_list))
        avgR2Lasso_values.append(np.mean(R2Lasso_list))
        avgR2Ridge_values.append(np.mean(R2Ridge_list))
        avgR2LinReg_values.append(np.mean(R2LinReg_list))
        
    minMSE1 = min(avgMSELasso_values)
    idx = avgMSELasso_values.index(minMSE1)
    minLmda = lmda_values[idx]
    LassoAvgR2 = avgR2Lasso_values[idx]
    
    minMSE2 = min(avgMSERidge_values)
    idx2 = avgMSERidge_values.index(minMSE2)
    minLmda2 = lmda_values[idx2]
    RidgeAvgR2 = avgR2Ridge_values[idx2]
    
    LinRegAvgR2 = np.mean(avgR2LinReg_values)

    y_train, y_test, X_train, X_test = train_test_split(tempDF[IV], tempDF[DVs], test_size=0.2)
    model1 = Lasso(alpha=minLmda, normalize=True)
    model1.fit(X_train, y_train)
    model2 = Ridge(alpha=minLmda2, normalize=True)
    model2.fit(X_train, y_train)
    model3 = LinearRegression(normalize=True)
    model3.fit(X_train, y_train)

        
        
    
    print(minLmda, minMSE1, model1.coef_, model1.intercept_, np.mean(avgR2Lasso_values))
    print(minLmda2, minMSE2, model2.coef_, model2.intercept_, np.mean(avgR2Ridge_values))
    print(model3.coef_, model3.intercept_, np.mean(avgR2LinReg_values))
        
    return lmda_values, avgMSELasso_values, avgMSERidge_values, avgMSELinReg_values, avgR2Lasso_values, avgR2Ridge_values, avgR2LinReg_values 


def plotMSEandR2vLambda(df, IV, DVs):
    lmda_values, avgMSELasso_values, avgMSERidge_values, avgMSELinReg_values, avgR2Lasso_values, avgR2Ridge_values, avgR2LinReg_values  = CrossValidateForMSEAtDiffLambda(df, IV, DVs)
    
    fig, axarr = plt.subplots(6, sharex=True)
    fig.set_size_inches(10, 7, forward=True)
    
    axarr[0].plot(lmda_values, avgMSELasso_values)
    axarr[0].set_title('Average MSE for k = 10 at different Lambda Values (Lasso)')
    #axarr[0].set_xlabel('Lambda')
    axarr[0].set_ylabel('Avg Test MSE')
    
    axarr[1].plot(lmda_values, avgMSERidge_values)
    axarr[1].set_title('Average MSE for k = 10 at different Lambda Values (Ridge)')
    #axarr[1].set_xlabel('Lambda')
    axarr[1].set_ylabel('Avg Test MSE')
    
    axarr[2].plot(lmda_values, avgMSELinReg_values)
    axarr[2].set_title('Average MSE for k = 10 at different Lambda Values (LinReg)')
    #axarr[2].set_xlabel('Lambda')
    axarr[2].set_ylabel('Avg Test MSE')
    
    axarr[3].plot(lmda_values, avgR2Lasso_values)
    axarr[3].set_title('Average R2 for k = 10 at different Lambda Values (Lasso)')
    #axarr[3].set_xlabel('Lambda')
    axarr[3].set_ylabel('Avg Test R2')
    
    axarr[4].plot(lmda_values, avgR2Ridge_values)
    axarr[4].set_title('Average R2 for k = 10 at different Lambda Values (Ridge)')
    #axarr[4].set_xlabel('Lambda')
    axarr[4].set_ylabel('Avg Test R2')
    
    axarr[5].plot(lmda_values, avgR2LinReg_values)
    axarr[5].set_title('Average R2 for k = 10 at different Lambda Values (LinReg)')
    axarr[5].set_xlabel('Lambda')
    axarr[5].set_ylabel('Avg Test R2')
    

