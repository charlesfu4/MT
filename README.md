# Automated Energy Profile Forecasting

Universal forecasting startegy on energy data with AutoML.

## Dataset

- [Smart meters in London](https://www.kaggle.com/jeanmidev/smart-meters-in-london)
- [Elia grid load](https://www.elia.be/en/grid-data/load-and-load-forecasts)
- More...

## AutoML libraries

- [TPOT](https://epistasislab.github.io/tpot/installing/)
- [Autosklearn](https://automl.github.io/auto-sklearn/master/installation.html)
- [H2O.ai](https://www.h2o.ai/)
- [Facebook Prophet](https://facebook.github.io/prophet/)

## Training pipeline

- Request local weather data from [Darksky API](https://darksky.net/dev)(Optional)

- Feature construnction with lag/ahead 

- Training process

_ Evaluation and confidence interval construction

### TPOT

1. Costomize your configuration of algorithm searching by modifying`tpot_multi.py`.

2. Train your model:
```
from tpot import TPOTRegressor
import tpot_multi
tpot_reg = TPOTRegressor(config_dict = tpot_multi)
tpot_reg.fit(train_X, train_y)
```
3. Time and space complexity issue emerges for multioutput regression with large number of features. See [Dask](https://examples.dask.org/machine-learning/tpot.html)
For more detail, see [Customizing TPOT](https://epistasislab.github.io/tpot/using/#customizing-tpots-operators-and-parameters)

### Autosklearn

- Support multioutput now, 

- A confidence interval method for regression estimator...

### H2O

- Not supporting multioutput regression

### Facebook Prophet

- Not so many experiments yet, support multioutput and confidence interval, but poor results.
