import matplotlib.lines as mlines
from numpy import arange, array, quantile

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_size_mul(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def plot_ci_forest(n, ci_term, estimator, features, targets):
    """confidence interval plot for forest estimator"""

    """
    Parameters
    ----------
    n: int
        The n timestamp of prediction going to be plotted

    ci_term:
    
    """
    predictions = []
    for est in estimator.estimators_:
        predictions.append(est.predict(features.iloc[n,:].to_numpy().reshape(1,-1)))
    predictions = np.array(predictions)
    prediction_list = predictions.reshape(predictions.shape[0], predictions.shape[2])
    fig = plt.figure(figsize=(16,7))
    plt.plot(np.quantile(prediction_list, 0.5, axis = 0), 'gx-', label='Prediction')
    plt.plot(np.quantile(prediction_list, ci_term, axis = 0), 'g--', label='{} % lower bond'.format(ci_term*100))
    plt.plot(np.quantile(prediction_list, 1-ci_term, axis = 0), 'g--', label='{} % upper bond'.format(100-ci_term*100))
    plt.plot(targets.iloc[n,:].to_numpy(), 'ro', label='Ground truth')
    plt.xlabel('hours', **font)
    plt.ylabel('KWh', **font)
    plt.legend(loc='upper left', fontsize = 15)
    plt.show()

## confidence interval check function for forest estimator
def verf_ci_qunatile_forest(quantile, estimator, ttest, pftest, n_samples):

    q_ub = []
    q_lb = []
    q_m = []
    for idx in range(n_samples):
        predictions = []
        for est in estimator.estimators_:
            predictions.append(est.predict(pftest.iloc[idx,:].to_numpy().reshape(1,-1)))
        predictions = np.array(predictions)
        prediction_list = predictions.reshape(predictions.shape[0], predictions.shape[2])
        q_ub.append(np.quantile(prediction_list, 1 - quantile, axis = 0))
        q_lb.append(np.quantile(prediction_list, quantile, axis = 0))
        q_m.append(np.quantile(prediction_list, 0.5, axis = 0))

    q_ub = np.array(q_ub)
    q_lb = np.array(q_lb)
    q_m = np.array(q_m)

    precentage_list = []
    err_count = 0
    for i in range(n_samples):
        count = 0
        for j in range(ttest.shape[1]):
            if ttest.iloc[i,j] >=  q_ub[i,j] or ttest.iloc[i,j] <= q_lb[i,j]:
                count += 1
        if count/ttest.shape[1] > quantile*2:
            err_count += 1
        precentage_list.append(count/ttest.shape[1])

    print("out_of_bound_pecentage", err_count/n_samples)
    fig = plt.figure(figsize = (16,7))
    font = {'family' : 'Lucida Grande',
            'weight' : 'bold',
            'size'   : 15}
    plt.rc('font', **font)
    plt.style.use('seaborn')
    plt.xlabel('Number of testing sets', **font)
    plt.ylabel('Out_of_bound_error', **font)
    plt.plot(precentage_list)



## confidence interval plot of n's prediction
def plot_conf_dynamic(predicted_error, test_y, ypred_t, n, ci_term):
    # confidence interval
    if(ci_term == 1.96):
        alpha = 5
    elif(ci_term == 1.645):
        alpha = 10
    elif(ci_term == 1.28):
        alpha = 20

    std = np.sqrt(predicted_error)
    ypred_t_ub = ypred_t + ci_term*std 
    ypred_t_lb = ypred_t - ci_term*std 
    # plot
    fig = plt.figure(figsize=(16,7))
    font = {'family' : 'Lucida Grande',
            'weight' : 'bold',
            'size'   : 15}
    plt.rc('font', **font)
    plt.style.use('seaborn')
    plt.plot(ypred_t[n, :].reshape(-1,1), 'gx-',label='Prediction')
    plt.plot(ypred_t_ub[n, :].reshape(-1,1), 'g--', label='{} % upper bond'.format(100-alpha*0.5))
    plt.plot(ypred_t_lb[n, :].reshape(-1,1), 'g--', label='{} % lower bond'.format(alpha*0.5))
    plt.plot(test_y.iloc[n, :].to_numpy().reshape(-1,1), 'ro', label='Ground truth')
    #plt.fill(np.concatenate([xx, xx[::-1]]),
    #         np.concatenate([y_upper, y_lower[::-1]]),
    #         alpha=.5, fc='b', ec='None', label='90% prediction interval')
    plt.xlabel('hours', **font)
    plt.ylabel('KWh', **font)
    plt.legend(loc='upper left', fontsize = 15)
    plt.show()

def plot_conf_static(val_y, val_y_pred, test_y, test_y_pred, n, ci_term):
    # confidence interval
    if(ci_term == 1.96):
        alpha = 5
    elif(ci_term == 1.645):
        alpha = 10
    elif(ci_term == 1.28):
        alpha = 20
        
    std = np.std(val_y - val_y_pred).to_numpy() 
    ypred_t_ub = test_y_pred  + ci_term*std 
    ypred_t_lb = test_y_pred  - ci_term*std 
    # plot
    fig = plt.figure(figsize=(16,7))
    font = {'family' : 'Lucida Grande',
            'weight' : 'bold',
            'size'   : 15}
    plt.rc('font', **font)
    plt.style.use('seaborn')
    plt.plot(test_y_pred[n, :].reshape(-1,1), 'gx-',label='Prediction')
    plt.plot(ypred_t_ub[n, :].reshape(-1,1), 'g--', label='{} % upper bond'.format(100-alpha*0.5))
    plt.plot(ypred_t_lb[n, :].reshape(-1,1), 'g--', label='{} % lower bond'.format(alpha*0.5))
    plt.plot(test_y.iloc[n, :].to_numpy().reshape(-1,1), 'ro', label='Ground truth')
    #plt.fill(np.concatenate([xx, xx[::-1]]),
    #         np.concatenate([y_upper, y_lower[::-1]]),
    #         alpha=.5, fc='b', ec='None', label='90% prediction interval')
    plt.xlabel('hours', **font)
    plt.ylabel('KWh', **font)
    plt.legend(loc='upper left', fontsize = 15)
    plt.show()

## Distribution plot funciton
def distri_plot(df):
    num_cols = df.shape[1]
    f, axes = plt.subplots(num_cols//3 + 1, 3, figsize=(15, 11), sharex=False)
    for idx, col_name in enumerate(df.columns, 0): 
        idx = int(idx)
        sns.distplot(df[col_name],ax=axes[idx//3,idx%3])  
    ## plot     
    plt.tight_layout()
    
## Scatter plot function
def scatter_plot(df):
    num_cols = df.shape[1]
    f, axes = plt.subplots(num_cols//2 + 1, 2, figsize=(15, 11), sharex=False)
    for idx, col_name in enumerate(df.columns, 0): 
        idx = int(idx)
        sns.scatterplot(x= col_name,y = "energy", data = df, ax=axes[idx//2,idx%2])  
    ## plot     
    plt.tight_layout()

## plot dataframe creation
def plot_df(arr, name):
    plot_df = pd.DataFrame()
    i = 0
    for row in arr:
        plot_df.insert(i, "{}".format(name), row, True) 
        i += 1
    return plot_df

def calibration_plot(observation, prediction, name, index = 0):
    """
    Parameters
    ----------
    observation: pandas.dataframe 
        The real value observered.
        
    prediction: list
        The list contain predictions with different esitmators in ensemble.
    
    name: str 
        The name for the plot: "calibration_{name}_{index}"
    
    index: int, default = 0
        Index assigned to be checked.
        
    """
    obs = observation.to_numpy()
    ci_range1 = arange(0.1, 1.,0.1)
    ci_range2 = arange(0.9, 1., 0.025)
    ci_range = np.concatenate([ci_range1, ci_range2], axis = 0)
    fraction = [] #empty list for storing fraction of obs within each CI bound
    for ci in ci_range:
        ub = quantile(prediction, 1-(1-ci)/2, axis = 0)[index,:]
        lb = quantile(prediction, (1-ci)/2, axis = 0)[index,:]
        # calculate fraction of points out of bound
        count = 0
        for idx in range(observation.shape[1]):
            if (obs[index,idx] <= ub[idx] and obs[index,idx] >= lb[idx]):
                count += 1
        fraction.append(count/observation.shape[1])
        
    
    
    # Plot the calibration plot
    plt.style.available
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=set_size(398))
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 11pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    }
    plt.rcParams.update(tex_fonts)
    
    # only these two lines are calibration curves
    plt.plot(fraction,ci_range, marker='o', linewidth=1)

    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    ax.add_line(line)
    fig.suptitle('Calibration plot for Confidence Interval')
    ax.set_xlabel('Contain percentage yielded')
    ax.set_ylabel('Expacted contain percentage')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    fig.savefig('../figures/london/calibration_{}_{}.png'.format(name, index), dpi= 300,
            format='png', bbox_inches='tight')
    
    
