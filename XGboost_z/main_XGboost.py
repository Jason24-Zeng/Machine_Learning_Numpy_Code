from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def main():
    Boston = load_boston()
    data = pd.DataFrame(data=Boston.data)
    data.columns = Boston.feature_names
    data['PRICE'] = Boston.target
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    xgb_reg = xgb.XGBRegressor(max_depth = 3, learning_rate=0.1, n_estimators=10, objective='reg:squarederror',
                               min_child_weight=5, random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("The root mean squared value is {}".format(rmse))

if __name__ == '__main__':
    main()