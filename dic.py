import numpy as np

# Check the TPOT documentation for information on the structure of config dicts

mul_reg_config_dict = {
    
    #regressors need not wrappers
    'sklearn.multioutput.RegressorChain': {
        'estimator': {
            'xgboost.XGBRegressor': {
                'n_estimators': [100],
                'max_depth': range(1, 11),
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'subsample': np.arange(0.05, 1.01, 0.05),
                'min_child_weight': range(1, 21),
                'nthread': [1],
                'objective': ['reg:squarederror']
            }
           
        
        }
    }

}
