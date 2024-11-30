import numpy as np

# 创建-1~1的2*10*100_0000的随机数组
random_array = np.random.uniform(-1, 1, (2, 10, 1000000))
random_array.shape  # Return the shape to confirm dimensions
print(random_array[0][0][0])