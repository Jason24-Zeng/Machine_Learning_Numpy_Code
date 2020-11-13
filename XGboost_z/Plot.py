from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.datasets import load_boston
#from sklearn.metrics import mean_squared_error
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt


def main():
    Boston = load_boston()
    data = pd.DataFrame(data=Boston.data)
    data.columns = Boston.feature_names
    data['PRICE'] = Boston.target
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    data_dmatrix = xgb.DMatrix(data = X, label = y)
    params = {"objective": "reg:squarederror", "learning_rate": 0.1, "colsample_bytree": 0.3, "max_depth": 5,
              "alpha": 10}
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=5)
    xgb.plot_tree(xg_reg, num_trees=1)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.savefig('./xgboost_testing1.jpg', dpi=1200, format='jpg')
    plt.show()
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [50, 5]
    plt.savefig('./xgboost_testing2.jpg', dpi=600, format='jpg')

    plt.show()

if __name__ == '__main__':
    main()
