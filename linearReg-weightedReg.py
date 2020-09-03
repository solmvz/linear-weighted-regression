import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error


def LinearReg(x_train, y_train, x_test, y_test):
    x_train = np.array(x_train)
    x_train = x_train.reshape(-1, 1)
    x_test = np.array(x_test)
    x_test = x_test.reshape(-1, 1)

    model = LinearRegression()
    #phi_train = np.concatenate((np.ones((len(x_train), 1)), x_train, np.power(x_train, 1)), axis=1)
    phi_train = np.concatenate((np.ones((len(x_train), 1)), x_train, np.power(x_train, 2), np.power(x_train, 3),
                    np.power(x_train, 4), np.power(x_train, 5), np.power(x_train, 6), np.power(x_train, 7),
                    np.power(x_train, 8), np.power(x_train, 9)), axis=1)
    model.fit(phi_train, y_train)
    #phi_test = np.concatenate((np.ones((len(x_test), 1)), x_test, np.power(x_test, 1)), axis=1)
    phi_test = np.concatenate((np.ones((len(x_test), 1)), x_test, np.power(x_test, 2), np.power(x_test, 3),
                  np.power(x_test, 4), np.power(x_test, 5), np.power(x_test, 6), np.power(x_test, 7),
                  np.power(x_test, 8), np.power(x_test, 9)), axis=1)
    y_pred = model.predict(phi_test)
    new_list = zip(x_test, y_pred)
    new_list = sorted(new_list)
    tuples = zip(*new_list)
    x_test, y_pred = [list(tuple) for tuple in tuples]

    plt.scatter(x_train, y_train, color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)
    plt.scatter(x_test, y_test, color='green')

    plt.show()

    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))

    return


def WeightedReg(x_train, y_train, x_test, y_test, tau=2):
    x_train = x_train.reshape(x_train.shape[0])
    y_train = y_train.reshape(y_train.shape[0])
    x_test = y_test.reshape(y_test.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    y_pred = []

    for i in range(len(x_test)):
        x0 = x_test[i]
        W = np.zeros((len(x_train), len(x_train)))
        new_X = np.ones((len(x_train), 2))

        for j in range(len(x_train)):
            new_X[j][1] = x_train[j] - x0
            W[j][j] = np.exp((x_train[j] - x0) ** 2 / -(2 * tau * tau))

        XW = np.matmul(new_X.T, W)
        pred = np.matmul(np.matmul(np.linalg.inv(np.matmul(XW, new_X)), XW), y_train)[0]
        # print(x_train.shape)
        # print(W.shape)
        # print(W.view())
        # print(x_train.view())
        y_pred.append(pred)

    new_list = zip(x_test, y_pred)
    new_list = sorted(new_list)
    tuples = zip(*new_list)
    x_test, y_pred = [list(tuple) for tuple in tuples]

    plt.scatter(x_train, y_train, color='black')
    plt.scatter(x_test, y_test, color='green')
    plt.scatter(x_test, y_pred, color='red', linewidth=2)

    plt.show()
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    return

def main():
    features = np.genfromtxt('features.txt')
    labels = np.genfromtxt('labels.txt')

    x = np.array(features)
    y = np.array(labels)

    x_train, x_test = x[:80], x[80:]
    y_train, y_test = y[:80], y[80:]

    #LinearReg(x_train, y_train, x_test, y_test)
    WeightedReg(x_train, y_train, x_test, y_test)


main()
