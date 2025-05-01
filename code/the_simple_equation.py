import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simple_equation(x,y):
    #最小二乘法(simple equation)
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # 这里我直接使用矩阵乘法简化了求和过程
    # 最原始的使用for循环遍历，很低效，可读性也很差
    # 其次是使用numpy的sum函数，效率高了很多
    # 最后我想到用numpy的矩阵乘法，效率最高，可取性也最高！
    xi_times_yi = x @ y
    xi_squared = x @ x

    w_hat = (xi_times_yi - n * x_mean * y_mean)/(xi_squared - n * (x_mean)**2)
    b_hat = y_mean - w_hat * x_mean
    return w_hat, b_hat



if __name__ == "__main__":
    training_set = pd.read_csv('linear_regression_data.csv')
    x = training_set['x'].values
    y = training_set['y'].values

    # 可视化生成的数据
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, label='Training Data')
    
    w_hat, b_hat = simple_equation(x,y)
    # 生成直线
    plt.plot(x,w_hat*x+b_hat, color = 'red')
    plt.show()