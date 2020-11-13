from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
Boston = load_boston()
#print(Boston.keys)
data = pd.DataFrame(data=Boston.data)
# print(Boston.DESCR)
# print(data)
data.columns = Boston.feature_names
data['PRICE'] = Boston.target
X, y = data.iloc[:, :-1], data.iloc[:, -1]
data_dmatrix = xgb.DMatrix(data = X, label= y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_test, y_test)
pred = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(pred, y_test))
print('root mean squared error: {}'.format(rmse))

