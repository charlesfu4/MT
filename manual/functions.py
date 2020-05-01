import seaborn as sns
import pandas as pd
import numpy as np
from pandas import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

## convert one to multiple series
def lag_ahead_series(data, n_in=1, n_out=1, n_vars = 1,dropnan=True):
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for j in range(n_vars):
        for i in range(n_in, 0, -1):
            cols.append(df.iloc[:,j].shift(i))
            names.append('var{}(t-{})'.format(j+1, i))
    
    # forecast sequence (t+1, ... t+n)
    for j in range(n_vars):
        for i in range(0, n_out):
            cols.append(df.iloc[:,j].shift(-i))
            names += [('var{}(t+{})'.format(j+1, i)) ]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

## Distribution plot funciton
def distri_plot(df):
    f, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=False)
    for idx, col_name in enumerate(df.columns, 0): 
        idx = int(idx)
        ## jump to plotting energy
        if(col_name == "LCLid"):
            sns.distplot(df["energy"],ax=axes[2,2])
            return
        sns.distplot(df[col_name],ax=axes[idx//3,idx%3])  
    ## plot     
    plt.tight_layout()

## Scatter plot function
def scatter_plot(df):
    f, axes = plt.subplots(4, 2, figsize=(15, 11), sharex=False)
    for idx, col_name in enumerate(df.columns, 0): 
        idx = int(idx)
        if(idx >= 8):
            return
        ## jump to plotting energy
        sns.scatterplot(x= col_name,y = "energy", data = df, ax=axes[idx//2,idx%2])  
    ## plot     
    plt.tight_layout()
    

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
                names.append('{}{}(t-{})'.format(df.columns[0], j+1, i))
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
                names.append('{}{}(t-{})'.format(df.columns[0], j+1, i))

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
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df.drop(['date'], axis = 1, inplace = True)
    return df

## feature/ target construction fucntion with lag variable
def feature_target_construct(df, load_lag, target_ahead, temp_lag, tskip = 0, wd_on = False, d_on = False, m_on = False, h_on = False, q_on = False):
    tempcols = ['temperature']
    load = df['energy']
    f_temp = pd.DataFrame()
    
    ## temp ahead series
    for col in tempcols:
        if(tskip != 0):
            temp = lag_ahead_series(df[col], 
                                 n_in = temp_lag,
                                 n_out = 0,
                                 n_vars = 1,
                                 dropnan = True,
                                 nskip= tskip)
        else:
             temp = lag_ahead_series(df[col], 
                                 n_in = temp_lag,
                                 n_out = 0,
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
        hour = pd.get_dummies(df['hour'])
        # alignment
        hour , f = hour.align(f, 'inner', axis = 0)
        f = pd.concat([hour, f], axis = 1)
    else:
        hour = df['hour']
        # alignment
        hour , f = hour.align(f, 'inner', axis = 0)
        f = pd.concat([hour, f], axis = 1)

    ## quarter one hot on
    if q_on:
        # month one hot encoding
        minute = pd.get_dummies(df['minute'])
        # alignment
        minute , f = minute.align(f, 'inner', axis = 0)
        f = pd.concat([minute, f], axis = 1)
    else:
        minute = df['minute']
        # alignment
        minute , f = minute.align(f, 'inner', axis = 0)
        f = pd.concat([minute, f], axis = 1)

    ## weekday one hot on
    if wd_on:
        # weekday one hot encoding
        weekday = pd.get_dummies(df['wd'])
        # alignment
        weekday , f = weekday.align(f, 'inner', axis = 0)
        f = pd.concat([weekday, f], axis = 1)
    else:
        weekday = df['wd']
        # alignment
        weekday , f = weekday.align(f, 'inner', axis = 0)
        f = pd.concat([weekday, f], axis = 1)
        
    ## day one hot on
    if d_on:
        # day one hot encoding
        day = pd.get_dummies(df['day'])
        # alignment
        day , f = day.align(f, 'inner', axis = 0)
        f = pd.concat([day, f], axis = 1)
    else:
        day = df['day']
        # alignment
        day , f = day.align(f, 'inner', axis = 0)
        f = pd.concat([day, f], axis = 1)
    
    ## month one hot on
    if m_on:
        # month one hot encoding
        month = pd.get_dummies(df['month'])
        # alignment
        month , f = month.align(f, 'inner', axis = 0)
        f = pd.concat([month, f], axis = 1)
    else:
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
## ci construction for err
def ci_construct(error, n):
    mean_list = []
    std_list = []
    for i in range(error.shape[1]):
        mean_list.append(np.mean(error.iloc[:,i]))
        std_list.append(np.std(error.iloc[:,i])*n)
    return mean_list, std_list
