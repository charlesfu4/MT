# Automatic Energy Profile Forecasting

Applying AutoML on forecasting energy profiles.

## Dataset

- Reference: [Smart meters in London](https://www.kaggle.com/jeanmidev/smart-meters-in-london)

## Environment setup

- [TPOT](https://epistasislab.github.io/tpot/installing/)
- [Autosklearn](https://automl.github.io/auto-sklearn/master/installation.html)
- NAS: Neural Network method, has yet to decide.

## Training pipeline

1. Data cleaning and merging among household electricy consumption and weather details

2. Construct costomized dictionary for multi-output regression

3. Train your model:
```
from tpot import TPOTRegressor
import tpot_multi
tpot_reg = TPOTRegressor(config_dict = tpot_multi)
tpot_reg.fit(train_X, train_y)
```

Costomize your configuration of algorithm searching by modifying`tpot_multi.py`.
For more detail, see [Customizing TPOT](https://epistasislab.github.io/tpot/using/#customizing-tpots-operators-and-parameters)


