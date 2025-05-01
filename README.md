# Linear Regression with The Simple Equation

<div align="center">
  <img src="./pictures/logo.png" alt="Project Logo" width="200">
  <br>
  <p>基于最小二乘法的线性回归实现</p>
</div>

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 项目亮点
### 创新性设计
- **自动生成训练集**：自动生成用于线性回归所需要的训练集
- **使用最小二乘法拟合线性回归**：利用最小二乘法直接求解出参数
- **算法创新**：项目中有不少的算法创新，我将一一展示

## 🔍 原理展示
### 数学推导可视化
#### 最小二乘法原理
对于数据点 $(x_i, y_i)$，我们寻找最佳拟合直线 $y = mx + b$，使得残差平方和最小：

$$
MINIMIZE \sum_{i=1}^{n}(y_i - (mx_i + b))^2
$$

#### 参数求解过程
    ```python
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
   
![Derivation Process](./assets/derivation_animation.gif)

## 🚀 创新性实现
### 与传统实现的对比
| 功能                | 传统实现          | 本项目的创新        |
|--------------------|-----------------|-------------------|
| 拟合方式            | 梯度下降          | 最小二乘法       |
| 可视化支持          | 无              | 实时拟合动画         |
| 异常处理            | 无              | 自动数据校验系统     |
| 计算复杂度          | O(n)            | O(1) 增量更新      |

### 核心技术突破
1. **使用矩阵乘法算出参数**：
   ```python
   # 这里我直接使用矩阵乘法简化了求和过程
   # 最原始的使用for循环遍历，很低效，可读性也很差
   # 其次是使用numpy的sum函数，效率高了很多
   # 最后我想到用numpy的矩阵乘法，效率最高，可取性也最高！
   xi_times_yi = x @ y
   xi_squared = x @ x
2. **最小二乘法算出参数**
   ```python
   w_hat = (xi_times_yi - n * x_mean * y_mean)/(xi_squared - n * (x_mean)**2)
   b_hat = y_mean - w_hat * x_mean
3. **梯度下降中的优化策略**
  ####1.使用矩阵乘法算梯度
   ```python
   dw = (1/m) * (y_pred - y) @ x
  ####2.记录loss值用于调整学习率
  ```python
     loss = (1/(2*m)) * np.sum((y_pred - y) ** 2)
     loss_history.append(loss)

