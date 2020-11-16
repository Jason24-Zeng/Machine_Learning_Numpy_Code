# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import SampleLoad
#from scipy.stats import binom
import IterationTrain


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = SampleLoad.Loadsample()
    print(train_data.iloc[0])
    #print(train_data)
    # contribution1 = binom.pmf(6, 10, 0.6)
    # contribution2 = binom.pmf(3, 10, 0.6)
    # print(contribution1, contribution2)
    theta = [0.2, 0.1]
    new_theta = IterationTrain.iterations_EM(data=train_data, theta=theta, iter=10)
    print(new_theta)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
