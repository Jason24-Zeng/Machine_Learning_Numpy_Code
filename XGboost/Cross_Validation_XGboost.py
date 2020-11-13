from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.datasets import load_boston
#from sklearn.metrics import mean_squared_error
import pandas as pd
#import numpy as np

def main():
    Boston = load_boston()
    data = pd.DataFrame(data=Boston.data)
    data.columns = Boston.feature_names
    data['PRICE'] = Boston.target
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    data_dmatrix = xgb.DMatrix(data = X, label = y)
    params = {"objective":"reg:squarederror", "learning_rate": 0.1, "colsample_bytree": 0.3, "max_depth": 5, "alpha": 10}
    # cv_results = xgb.cv(params=params, dtrain=data_dmatrix, num_boost_round=10, nfold=3, as_pandas=True, metrics="rmse",
    #                     seed=42, early_stopping_rounds=10)
    #params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                        num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

    print(cv_results.head(5), cv_results.tail(3))

if __name__ == '__main__':
    main()