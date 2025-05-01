import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


np.random.seed(123)


num_samples = 100  
true_w = 2.5       
true_b = 1.7       
noise_scale = 0.8  


x = np.random.rand(num_samples) * 10  # 生成0-10之间的均匀分布数据


y = true_w * x + true_b + np.random.randn(num_samples) * noise_scale

# 可视化生成的数据
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label='Training Data')
plt.plot(x, true_w * x + true_b, color='red', linewidth=2, label='True Relationship')
plt.xlabel('X - Input Feature')
plt.ylabel('y - Target Value')
plt.title('Synthetic Linear Regression Data')
plt.legend()
plt.grid(True)
plt.show()

# 打印前5个样本查看
print("Generated data samples (first 5 rows):")
for i in range(5):
    print(f"x = {x[i]:.2f}, y = {y[i]:.2f}")


np.savetxt('linear_regression_data.csv', np.column_stack((x, y)), delimiter=',', header='x,y', comments='')