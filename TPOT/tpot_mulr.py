import numpy as np

# Check the TPOT documentation for information on the structure of config dicts

mul_reg_config_dict = {
    
    #regressors need to be wrapped by multioutput

    'sklearn.multioutput.RegressorChain': {
        'estimator': {
            'xgboost.XGBRegressor': {
                'n_estimators': [100],
                'learning_rate': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.],
                'max_depth': range(1, 15),
            }
        }
    },


    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.03, 1.01, 0.03),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },





}
