import seaborn as sns
import pandas as pd
import numpy as np
from pandas import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.utils import resample

# bootstrapping confidence interval
def bootstrapping_ci(train_X, train_Y, test_X, estimator, n_bootstraps = 100,  p_samples = 0.5):
    """
    Parameters
    ----------
    train_X: pd.DataFrame or numpy.array
        The original features of training set
        
    train_Y: pd.DataFrame or numpy.array
        The original targets of training set
        
    test_X: pd.DataFrame or numpy.array
        The original features of testing set
        
    estimator: sklearn.BaseEstimator
        The estimator to fit and for generating prediction
    
    n_bootstraps: int, default = 100
        The number of resamples taken by bootstrapping
    
    p_samples: float, default = 0.5
        The proportion of random sampling takes from original training data
        
    """
    
    b_x = []
    b_y = []
    prediction_bs = []
    for _ in range(n_bootstraps):
        sample_X ,sample_y = resample(train_X, train_Y, n_samples = int(trian_X.shape[0]*p_samples))
        b_x.append(sample_X)
    b_y.append(sample_y)
    """Now fit the estimators and generate different predictions n_bootstraps times"""
    for i, feature in enumerate(b_x):
        estimator.fit(feature, b_y[i])
        prediction_bs.append(estimator.predict(pftest))
    
    return prediction_bs
## Distribution transformation
def power_trans(df):
    pt = PowerTransformer(method = "yeo-johnson")
    pt.fit(df)
    df_trans = pd.DataFrame(pt.transform(df), columns=train.columns[:])
    return df_trans


def quantile_trans(df, n):
    rng = np.random.RandomState(304)
    qf = QuantileTransformer(n_quantiles=n, output_distribution='normal',random_state=rng)
    qt.fit(df)
    df_trans = pd.DataFrame(qt.transform(df), columns=train.columns)
    return df_trans

## convert one to multiple series
def lag_ahead_series(data, n_in=1, n_out=1, n_vars = 1, dropnan=True, nskip = 0):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    if n_in + n_out == 0:
        return
    # skipped sequences (t + nskip, ... t + n)
    if nskip != 0:
        # input sequence (t-n, ..., t-1) 
        for j in range(n_vars):
            for i in range(n_in, 0, -nskip):
                cols.append(df.iloc[:,j].shift(i))
                names.append('{}{}(t-{})'.format(df.columns[j], j+1, i))
        # forecast sequence (t+1, ..., t+n)
        for j in range(n_vars):
            for i in np.arange(0, n_out, nskip):
                cols.append(df.iloc[:,j].shift(-i))
                names += [('{}{}(t+{})'.format(df.columns[j], j+1, i))]
    # regular sequences
    else:
        # input sequence (t-n, ... t-1)
        for j in range(n_vars):
            for i in range(n_in, 0, -1):
                cols.append(df.iloc[:,j].shift(i))
                names.append('{}{}(t-{})'.format(df.columns[j], j+1, i))

        # forecast sequence (t+1, ... t+n)
        for j in range(n_vars):
            for i in range(0, n_out):
                cols.append(df.iloc[:,j].shift(-i))
                names += [('{}{}(t+{})'.format(df.columns[j], j+1, i))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

## convert one to multiple series
def tfla_series(data, n_in=1, n_out=1, n_vars = 1, dropnan=True, nskip = 0):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    if n_in + n_out == 0:
        return
    # skipped sequences (t + nskip, ... t + n)
    if nskip != 0:
        # input sequence (t-n, ..., t-1) 
        for j in range(n_vars):
            for i in range(n_in, 0, -nskip):
                cols.append(df.iloc[:,j].shift(i))
                names.append('{}'.format(-i+n_in))
        # forecast sequence (t+1, ..., t+n)
        for j in range(n_vars):
            for i in np.arange(0, n_out, nskip):
                cols.append(df.iloc[:,j].shift(-i))
                names += [('{}{}(t+{})'.format(df.columns[0], j+1, i))]
    # regular sequences
    else:
        # input sequence (t-n, ... t-1)
        for j in range(n_vars):
            for i in range(n_in, 0, -1):
                cols.append(df.iloc[:,j].shift(i))
                names.append('{}'.format(-i+n_in))

        # forecast sequence (t+1, ... t+n)
        for j in range(n_vars):
            for i in range(0, n_out):
                cols.append(df.iloc[:,j].shift(-i))
                names += [('{}{}(t+{})'.format(df.columns[0], j+1, i))]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def tf_construct(df, target_name, load_lag, target_ahead):
    load = df['%s'% target_name]
    ## load lag series
    f = tfla_series(load,
                  n_in = load_lag,
                  n_out = 0,
                  n_vars = 1,
                  dropnan = True)
    ## target part
    t = tfla_series(load,
                  n_in = 0,
                  n_out = target_ahead,
                  n_vars = 1,
                  dropnan = True)
    # alignment of feature and target
    f, t = f.align(t, 'inner', axis = 0)
    
    return f, t

## plot dataframe creation
def plot_df(arr, name):
    plot_df = pd.DataFrame()
    i = 0
    for row in arr:
        plot_df.insert(i, "{}".format(name), row, True) 
        i += 1
    return plot_df

def get_eval(y, yhat):
    print("MSE: {}".format(mean_squared_error(y,yhat)))
    print("MAE: {}".format(mean_absolute_error(y,yhat)))
    print("r2_score: {}".format(r2_score(y,yhat, multioutput = "variance_weighted")))
    
## extract day and month
def extract_dmhq(df):
    df['date'] = df.index.astype('datetime64[ns]')
    df['wd'] = df['date'].dt.dayofweek 
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['wd'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['yd'] = df['date'].dt.dayofyear
    df.drop(['date'], axis = 1, inplace = True)
    return df

## feature/ target construction fucntion with lag variable
def feature_target_construct(df, target_name,load_lag, target_ahead, temp_lag, temp_ahead, f_picked, tskip = 0, wd_on = False, d_on = False, m_on = False, h_on = False, q_on = False):
    tempcols = f_picked 
    load = df['%s'% target_name]
    f_temp = pd.DataFrame()
    
    ## temp ahead series
    for col in tempcols:
        if(tskip != 0):
            temp = lag_ahead_series(df[col], 
                                 n_in = temp_lag,
                                 n_out = temp_ahead,
                                 n_vars = 1,
                                 dropnan = True,
                                 nskip= tskip)
        else:
             temp = lag_ahead_series(df[col], 
                                 n_in = temp_lag,
                                 n_out = temp_ahead,
                                 n_vars = 1,
                                 dropnan = True)
        f_temp = pd.concat([f_temp, temp], axis = 1)            
        
    ## load lag series
    f_load = lag_ahead_series(load,
                          n_in = load_lag,
                          n_out = 0,
                          n_vars = 1,
                          dropnan = True)
    # when f_temp exist
    if f_temp.shape[1] > 0:
        f_load, f_temp = f_load.align(f_temp, 'inner', axis = 0)
        f = pd.concat([f_temp, f_load], axis = 1)
    # when no f_temp
    else:
        f = f_load
        
    ## hour one hot on
    if h_on:
        # month one hot encoding
        hour = df['hour']
        # alignment
        hour , f = hour.align(f, 'inner', axis = 0)
        f = pd.concat([hour, f], axis = 1)

    ## quarter one hot on
    if q_on:
        # month one hot encoding
        minute = df['minute']
        # alignment
        minute , f = minute.align(f, 'inner', axis = 0)
        f = pd.concat([minute, f], axis = 1)

    ## weekday one hot on
    if wd_on:
        # weekday one hot encoding
        weekday = df['wd']
        # alignment
        weekday , f = weekday.align(f, 'inner', axis = 0)
        f = pd.concat([weekday, f], axis = 1)
        
    ## day one hot on
    if d_on:
        # day one hot encoding
        day = df['day']
        # alignment
        day , f = day.align(f, 'inner', axis = 0)
        f = pd.concat([day, f], axis = 1)
    
    ## month one hot on
    if m_on:
        # month one hot encoding
        month = df['month']
        # alignment
        month , f = month.align(f, 'inner', axis = 0)
        f = pd.concat([month, f], axis = 1)
        
        
    ## number of LCLid on                  
    if('LCLid' in df.columns):
        nlclid = df['LCLid']
        # alignment
        nlclid, f = nlclid.align(f, 'inner', axis = 0)       
        f = pd.concat([f, nlclid], axis = 1)
        
        
    ## target part
    t = lag_ahead_series(load,
                          n_in = 0,
                          n_out = target_ahead,
                          n_vars = 1,
                          dropnan = True)
    # alignment of feature and target
    f, t = f.align(t, 'inner', axis = 0)
    
    return f, t
