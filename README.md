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
1. 计算均值：
   $$\bar{x} = \frac{1}{n}\sum x_i,\quad \bar{y} = \frac{1}{n}\sum y_i$$
2. 计算w：
   $$ w = \frac{\sum_{i=0}^n(\bar{x_i}\bar{y_i}) - n\bar{x}\bar{y}}{\sum_{i=0}^n(\bar{x_i^2} - n\bar{x_i}^2)}
![Derivation Process](./assets/derivation_animation.gif)

## 🚀 创新性实现
### 与传统实现的对比
| 功能                | 传统实现          | 本项目的创新        |
|--------------------|-----------------|-------------------|
| 计算方式            | 批量计算          | 增量式更新（支持流式数据）|
| 可视化支持          | 无              | 实时拟合动画         |
| 数值稳定性          | 基础实现          | 添加正则化项         |
| 异常处理            | 无              | 自动数据校验系统     |
| 计算复杂度          | O(n)            | O(1) 增量更新      |

### 核心技术突破
1. **流式数据处理**：
   ```python
   # 增量更新参数
   def partial_fit(self, x, y):
       self.n += 1
       dx = x - self.mean_x
       self.mean_x += dx / self.n
       self.mean_y += (y - self.mean_y) / self.n
       self.cov += dx * (y - self.mean_y)
       self.var_x += dx * (x - self.mean_x)
       self.slope = self.cov / self.var_x
       self.intercept = self.mean_y - self.slope * self.mean_x
