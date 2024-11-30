import 神经网络
import 梯度下降算法
import image_to_array
import 求导
import numpy as np
import Sigmoid算法


image_array = image_to_array.image_to_array('1.jpg')

# 创建2*10*1000000的随机数组作为权重
random_array = np.random.uniform(-1, 1, (2, 10, 1000000))

random_array_result = np.random.rand(2, 10)
print(神经网络.NeuralNetwork(image_array,random_array,random_array_result).a_net_result())

