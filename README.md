# Automatic Energy Profile Forecasting

Applying AutoML on forecasting energy profiles.

## Dataset

- [Smart meters in London](https://www.kaggle.com/jeanmidev/smart-meters-in-london)
- [Elia grid load](https://www.elia.be/en/grid-data/load-and-load-forecasts)

## AutoML libraries

- [TPOT](https://epistasislab.github.io/tpot/installing/)
- [Autosklearn](https://automl.github.io/auto-sklearn/master/installation.html)
- [H2O.ai](https://www.h2o.ai/)

## Training pipeline

- Request local weather data from [Darksky API](https://darksky.net/dev)

- Data cleaning and concatenation of household electricy consumption and weather data
### TPOT

1. Costomize your configuration of algorithm searching by modifying`tpot_multi.py`.

2. Train your model:
```
from tpot import TPOTRegressor
import tpot_multi
tpot_reg = TPOTRegressor(config_dict = tpot_multi)
tpot_reg.fit(train_X, train_y)
```
For more detail, see [Customizing TPOT](https://epistasislab.github.io/tpot/using/#customizing-tpots-operators-and-parameters)

### Autosklearn

- Not supporting multioutput regression(working on it)

### H2O

- Not supporting multioutput regression
