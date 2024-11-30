import numpy as np
import image_to_array
import Sigmoid算法

class NeuralNetwork:
    def __init__(self,image_array,net,result_net):
        # 初始化,传入参数
        self.image_array = image_array
        self.a_net = net[0] # 神经网络第一层(a_net)的权重
        self.b_net = net[1] # 神经网络第二层(b_net)的权重
        self.result_net = result_net # 神经网络结果层(result_net)的权重
        print("初始化成功")
    def a_net_result(self):
        # 计算神经网络第一层(a_net)的结果
        self.a_net_0 = np.sum(self.image_array * self.a_net[0]) # 计算神经网络第一层第0节点(a_net_0)的结果
        self.a_net_1 = np.sum(self.image_array * self.a_net[1]) # 计算神经网络第一层第1节点(a_net_1)的结果
        self.a_net_2 = np.sum(self.image_array * self.a_net[2]) # 计算神经网络第一层第2节点(a_net_2)的结果
        self.a_net_3 = np.sum(self.image_array * self.a_net[3]) # 计算神经网络第一层第3节点(a_net_3)的结果
        self.a_net_4 = np.sum(self.image_array * self.a_net[4]) # 计算神经网络第一层第4节点(a_net_4)的结果
        self.a_net_5 = np.sum(self.image_array * self.a_net[5]) # 计算神经网络第一层第5节点(a_net_5)的结果
        self.a_net_6 = np.sum(self.image_array * self.a_net[6]) # 计算神经网络第一层第6节点(a_net_6)的结果
        self.a_net_7 = np.sum(self.image_array * self.a_net[7]) # 计算神经网络第一层第7节点(a_net_7)的结果
        self.a_net_8 = np.sum(self.image_array * self.a_net[8]) # 计算神经网络第一层第8节点(a_net_8)的结果
        self.a_net_9 = np.sum(self.image_array * self.a_net[9]) # 计算神经网络第一层第9节点(a_net_9)的结果
        self.a_net_result_number = Sigmoid算法.sigmoid(np.array([self.a_net_0,self.a_net_1,self.a_net_2,self.a_net_3,self.a_net_4,self.a_net_5,self.a_net_6,self.a_net_7,self.a_net_8,self.a_net_9]))
        return self.a_net_result_number
    def b_net_result(self):
        # 计算神经网络第二层(b_net)的结果
        self.a_net_result1 = self.a_net_result()
        self.b_net_0 = np.sum(self.a_net_result1[0] * self.b_net[0]) # 计算神经网络第二层第0节点(a_net_0)的结果
        self.b_net_1 = np.sum(self.a_net_result1[1] * self.b_net[1]) # 计算神经网络第二层第1节点(a_net_1)的结果
        self.b_net_2 = np.sum(self.a_net_result1[2] * self.b_net[2]) # 计算神经网络第二层第2节点(a_net_2)的结果
        self.b_net_3 = np.sum(self.a_net_result1[3] * self.b_net[3]) # 计算神经网络第二层第3节点(a_net_3)的结果
        self.b_net_4 = np.sum(self.a_net_result1[4] * self.b_net[4]) # 计算神经网络第二层第4节点(a_net_4)的结果
        self.b_net_5 = np.sum(self.a_net_result1[5] * self.b_net[5]) # 计算神经网络第二层第5节点(a_net_5)的结果
        self.b_net_6 = np.sum(self.a_net_result1[6] * self.b_net[6]) # 计算神经网络第二层第6节点(a_net_6)的结果
        self.b_net_7 = np.sum(self.a_net_result1[7] * self.b_net[7]) # 计算神经网络第二层第7节点(a_net_7)的结果
        self.b_net_8 = np.sum(self.a_net_result1[8] * self.b_net[8]) # 计算神经网络第二层第8节点(a_net_8)的结果
        self.b_net_9 = np.sum(self.a_net_result1[9] * self.b_net[9]) # 计算神经网络第二层第9节点(a_net_9)的结果
        self.b_net_result_number = Sigmoid算法.sigmoid(np.array([self.b_net_0,self.b_net_1,self.b_net_2,self.b_net_3,self.b_net_4,self.b_net_5,self.b_net_6,self.b_net_7,self.b_net_8,self.b_net_9]))
        return  self.b_net_result_number
    def result(self):
        # 计算神经网络结果层(result_net)的结果
        b_net_results = self.b_net_result()
        result_1_number = np.sum(b_net_results[0] * self.result_net[0]) # 计算神经网络结果层第0节点(result_0)的结果
        result_2_number = np.sum(b_net_results[1] * self.result_net[1]) # 计算神经网络结果层第1节点(result_1)的结果
        result_number = Sigmoid算法.sigmoid(np.array([result_1_number, result_2_number]))
        return result_number


