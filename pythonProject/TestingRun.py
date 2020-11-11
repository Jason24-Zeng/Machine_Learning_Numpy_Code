from sklearn.datasets import load_iris
#iris = load_iris()
import DecisionTree as dt
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
def main():
    iris = load_iris()
    features = iris.data[:, :2]
    labels = iris.target
    X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state= 24)

    decision_tree = dt.DecisionTree(_max_length = 4, _min_splits = 30)
    decision_tree.fit(X_train, y_train)
    predicted_label = decision_tree.predict(X_test)


    sk_decision_tree = DecisionTreeClassifier(max_depth= 4, min_samples_split = 30, random_state= 24)
    sk_decision_tree.fit(X_train, y_train)
    predicted_sk_label = sk_decision_tree.predict(X_test)

    print('accuracy_of_our_model:{0}'.format(accuracy(predicted_label, y_test)))
    print('accuracy_of_Scikit_Learn_model:{0}'.format(accuracy(predicted_sk_label, y_test)))


def accuracy(prediction, actual):
    """
    :param prediction:
    :param actual:
    :return accuaracy:

    Simple function to compute raw accuaracy score quick comparision.
    """
    correct_count = 0
    prediction_len = len(prediction)
    for idx in range(prediction_len):
        if int(prediction[idx]) == actual[idx]:
            correct_count += 1
    return correct_count/prediction_len
if __name__ == '__main__':
    main()
# feature = iris.data[:,:4]
# label = iris.target

# X_train, X_test, y_train, y_test = train_test_split(feature, label, random_state=42)
# decision_tree_model = dt.DecisionTree( 2, _min_splits = 30)
# decision_tree_model.fit(X_train, y_train)
# prediction  = decision_tree_model.predict(X_test)
# print("Our Model Accuracy : {0}".format(accuracy(prediction, y_test)))




