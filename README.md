# Linear Regression with The Simple Equation
# And the Comparation with The Gredient Dencent

<div align="center">
  <img src="./code/pictures/logo.png" alt="Project Logo" width="200">
  <br>
  <p>基于最小二乘法的线性回归实现</p>
</div>

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# 我的科技节作品

## 🎯 项目亮点
### 课本中的知识复现
高中数学必修2中提到的最小二乘法的代码实现。
### 创新性设计
- **自动生成训练集**：自动生成用于线性回归所需要的训练集
- **使用最小二乘法拟合线性回归**：利用最小二乘法直接求解出参数
- **算法创新**：项目中有不少的算法创新，我将一一展示
- **动态化展示**动态过程的梯度下降拟合，更直观凸显最小二乘法的高效

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
    ```
![Derivation Process](./assets/derivation_animation.gif)

## 🚀 创新性实现
### 与传统实现的对比
| 功能                | 传统实现          | 本项目的创新        |
|--------------------|-----------------|-------------------|
| 拟合方式            | 梯度下降          | 最小二乘法       |
| 可视化支持          | 无              | 实时拟合动画         |
| 计算复杂度          | O(n)            | O(1)       |

### 核心技术突破
1. **使用矩阵乘法算出参数**：
   ```python
   # 这里我直接使用矩阵乘法简化了求和过程
   # 最原始的使用for循环遍历，很低效，可读性也很差
   # 其次是使用numpy的sum函数，效率高了很多
   # 最后我想到用numpy的矩阵乘法，效率最高，可取性也最高！
   xi_times_yi = x @ y
   xi_squared = x @ x
   ```
2. **最小二乘法算出参数**
   ```python
   w_hat = (xi_times_yi - n * x_mean * y_mean)/(xi_squared - n * (x_mean)**2)
   b_hat = y_mean - w_hat * x_mean
3. **梯度下降中的优化策略**
   使用矩阵乘法算梯度,以及记录loss来调整学习率
   ```python
   dw = (1/m) * (y_pred - y) @ x

   -------------------------------------------
   loss = (1/(2*m)) * np.sum((y_pred - y) ** 2)
   loss_history.append(loss)
   ```
4.动态展示梯度下降过程
使用FuncAnimation()函数实现
    ```python
    
    def update(frame):
    # 更新回归线
    current_w = w_history[frame]
    current_b = b_history[frame]
    line_gd.set_data(x, current_w * x + current_b)
    # 绘制最小二乘线（保持固定）
    if frame == 0:
        line_ls.set_data(x, w_ls * x + b_ls)
    # 更新损失曲线
    cost_line.set_data(np.arange(frame), loss_history[:frame])
    return line_gd, line_ls, cost_line
    # 创建动画
    ani = FuncAnimation(fig, update, frames=range(n_iterations),
                    interval=50, blit=True, repeat=True)
                    
    ```

## 💻实现过程  
1.📄创建数据集
使用项目内置的traing_set_builder文件创建数据集
<div align = 'center'>
  <img src = './code/pictures/Figure_1.png'>
  <p>数据集</p>
</div>
2.🧮最小二乘法拟合
<div align = 'center'>
  <img src = './code/pictures/最小二乘法拟合的直线.png'>
  <p>拟合结果</p>
</div>
3.🔧梯度下降拟合
<div align = 'center'>
  <img src = './code/pictures/gredient_decent.png'>
  <p>拟合结果</p>
</div>
4.🔬动态比较
在这里我又创建了一个更大的数据集
<div align = 'center'>
  <img src = './code/pictures/Comparasion.png'>
  <p>大数据集</p>
</div>
动态比较
<div align = 'center'>
  <img src = './code/pictures/repeat.png'>
  <p>动态比较</p>
</div>


## 😀经验分享
### 梯度下降中的记录loss来调整学习率

#### 正常的α
<div align = 'center'>
  <img src = './code/pictures/loss_history.png'>
  <p>$$正常的\alpha$$</p>
</div>
#### 如果α太小
<div align = "center">
  <img src = "./code/pictures/太小.png">
  <p>$$\alpha太小$$</p>
</div>

#### 如果α太大
<div align = 'center'>
  <img src = './code/pictures/太大.png'>
  <p>$$\alpha太大$$</p>
</div>


# 🧪本地运行比较结果
1.确保您的设备上有python环境
[Python官网](https://www.python.org/)
2.确保您的python安装了必要的库
  ```python
  pip install numpy matplotlib pandas
  ```
3.克隆项目
  ```bash
  git clone https://github.com/13782036813/The-code-implementation-of-the-simlpe-equation-And-the-comparation-with-Gredient-Decent.git
  ```
4.运行
 确保您进入了code目录下
 ```bash
 python compartaor.py
 ```


