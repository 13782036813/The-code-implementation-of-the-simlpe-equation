import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gradient_descent(x, y, w, b, learning_rate=0.01, num_iterations=1000, loss_history=[], w_history=[], b_history=[]):
    m = len(x)
    for i in range(num_iterations):
        y_pred = w * x + b
        loss = (1/(2*m)) * np.sum((y_pred - y) ** 2)
        loss_history.append(loss)
        dw = (1/m) * (y_pred - y) @ x
        db = (1/m) * np.sum(y_pred - y)
        w -= learning_rate * dw
        b -= learning_rate * db
        w_history.append(w)
        b_history.append(b)
        # 为了让图像重复拟合 我要执行一下操作
        #if i % 300 == 0:
        #   w = b = 0
    return w, b, loss_history, w_history, b_history

if __name__ == "__main__":
    x = pd.read_csv('linear_regression_data.csv')['x'].values
    y = pd.read_csv('linear_regression_data.csv')['y'].values

    w = b = 0
    learning_rate = 0.01
    num_iterations = 1000
    # 执行梯度下降算法
    loss_history = []
    w, b, loss_history, w_history, b_history= gradient_descent(x, y, w, b, learning_rate, num_iterations,loss_history)
    # 绘制损失函数曲线

    plt.plot(loss_history, label='Loss Function')
    plt.grid(True)
    plt.show()