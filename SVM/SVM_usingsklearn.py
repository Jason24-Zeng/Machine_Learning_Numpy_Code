from sklearn import svm
from preprocessing import loadDataset
import matplotlib.pyplot as plt
import numpy as np

model = svm.SVC(kernel='rbf', C=10, gamma=1)
filename_train = 'train.txt'
X_train, y_train = loadDataset(filename=filename_train)
model.fit(X_train,y_train)
model.score(X_train,y_train)
filename_test = 'test.txt'
X_test, y_test = loadDataset(filename=filename_test)
# X_pos, X_neg = X_train[y_train == 1], X_train[y_train == -1]
# y_pos, y_neg = y_train[y_train == 1], y_train[y_train == -1]
# print(X_pos.shape, X_neg.shape, y_pos.shape, y_neg.shape)
# x_line = np.linspace(-0.5, 0.5, 100)
# y_line = 1 - x_line
X_pos, X_neg = X_test[y_test == 1], X_test[y_test == -1]
h = 0.002
x_min, x_max = - 0.7, 0.7
y_min, y_max = -0.7, 0.7
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.scatter(x=X_pos[0], y=X_pos[1], marker='o', s=100, linewidths=3)
plt.scatter(x=X_neg[0], y=X_neg[1], marker='x', s=100, linewidths=3)
plt.contour(xx, yy, Z, cmap= plt.cm.ocean, alpha=0.6)

plt.savefig('test_case.jpg')
plt.show()



predicted = model.predict(X_test)

# print(predicted)
# print(y_test)
#, y_test, predicted == y_test
def accuracy(y_pred, y_test):
    total_sample = len(y_pred)
    acc = float((y_pred == y_test).sum()/total_sample)
    return acc
print(accuracy(y_pred=predicted, y_test=y_test))
