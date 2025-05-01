from with_gredient_decent import gradient_descent
from the_simple_equation import simple_equation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = pd.read_csv("linear_regression_large_data.csv")['x'].values
y = pd.read_csv('linear_regression_large_data.csv')['y'].values


# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 我这里直接直译吧 最小二乘法->least squre->ls
w_ls, b_ls = simple_equation(x=x,y=y)

n_iterations = 150
loss_history = []
w_history = []
b_history = []
w_g = b_g = 0
w_g, b_g, loss_history, w_history, b_history = gradient_descent(x=x,y=y,w=w_g,b=b_g,num_iterations=n_iterations,loss_history=loss_history, w_history=w_history,b_history=b_history,learning_rate=0.001)

# 初始化绘图元素
ax1.scatter(x, y, color='blue', alpha=0.5, label='Data')
line_ls, = ax1.plot([], [], color='green', linewidth=2, label='simple equation')
line_gd, = ax1.plot([], [], color='red', linewidth=2, label='gredient decent')
ax1.set_xlim(min(x)-1, max(x)+1)
ax1.set_ylim(min(y)-1, max(y)+1)
ax1.legend()
ax1.set_title('COMPARASION')

ax2.set_xlim(0, n_iterations)
ax2.set_ylim(min(loss_history)-5, max(loss_history)+5)
cost_line, = ax2.plot([], [], color='orange')
ax2.set_xlabel('ITERATIONS')
ax2.set_ylabel('LOSS')
ax2.set_title('LOSS EVOLUTION')

# 动画更新函数
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

plt.tight_layout()
plt.show()