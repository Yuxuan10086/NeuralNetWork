import numpy as np

# 在函数之间交互时一律使用二维矩阵

class NeuralNetWork:
    def __init__(self, nodes, learn_rate):
        self.nodes = nodes
        self.weight = []
        self.tier = len(nodes)
        # for i in range(self.tier):
        for i in range(len(nodes) - 1):
            weight = np.random.normal(0.0, pow(nodes[i], -0.5), (nodes[i + 1], nodes[i]))
            self.weight.append(weight)
        self.final_outputs = []
        self.sigmoid = lambda x: 1 / (1 + pow(2.718, -x))
        self.learn_rate = learn_rate
        self.outputs_list = []

    def query(self, inputs):  # 传入输入列表
        inputs = np.array(inputs, ndmin = 2).T
        self.outputs_list.append(inputs)
        for i in range(self.tier - 1):
            inputs = np.dot(self.weight[i], inputs)
            for i in range(len(inputs)):
                inputs[i] = self.sigmoid(inputs[i])
            self.outputs_list.append(np.array(inputs, ndmin = 2))
        self.final_outputs = self.outputs_list[-1]
        # print(self.final_outputs)

    def train(self, inputs, targets): #传入列表
        self.query(inputs)
        # inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        errors = [self.final_outputs - targets]
        for i in range(self.tier - 1):
            errors = [np.dot(self.weight[-i - 1].T, errors[-i - 1])] + errors
        # print('errors', errors)
        for j in range(self.tier - 1):
            middle_entry = []
            for i in range(self.nodes[j + 1]):
                middle_entry.append(errors[j + 1][i] * self.outputs_list[j + 1][i] * (1 - self.outputs_list[j + 1][i]))
            middle_entry = np.array(middle_entry, ndmin = 2)
            # print('mid', middle_entry)
            derta_weight = self.learn_rate * middle_entry * self.outputs_list[j].T
            # print(derta_weight)
            self.weight[j] -= derta_weight
        # print('wei', self.weight)


    def check(self):
        print(self.nodes, 'weight', self.weight, self.tier, self.weight[0].shape)  #, "out", self.outputs_list)

# test = NeuralNetWork([5, 300, 10], 0.05)
# test.check()
# while 1:
#     test.query([0.7, 0.75, 0.8, 0.3, 0.5])
#     test.train([0.7, 0.75, 0.8, 0.3, 0.5], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
#     print(test.final_outputs)
# print('out', test.outputs_list, test.final_outputs.shape)
