import numpy as np
class network:
    def __init__(self, nodes, learn_rate): # nodes为列表,储存各层节点数
        self.nodes = nodes
        self.weight = []
        self.tier = len(nodes)
        # for i in range(self.tier):
        for i in range(len(nodes) - 1):
            weight = np.random.normal(0.0, pow(nodes[i], -0.5), (nodes[i], nodes[i]))
            self.weight.append(weight)
        self.final_outputs = []
        self.sigmoid = lambda x : 1 / (1 + pow(2.7182818, -x))
        self.learn_rate = learn_rate
        self.outputs_list = []

    def query(self, inputs): # 传入输入列表
        inputs = np.transpose(inputs)
        self.outputs_list.append(inputs)
        for i in range(self.tier - 1):
            # print(inputs)
            # print(self.weight)
            inputs = np.dot(self.weight[i], inputs)
            # print('inputs : ', inputs)
            for i in range(len(inputs)):
                inputs[i] = self.sigmoid(inputs[i])
            self.outputs_list.append(inputs)
        self.final_outputs = inputs
        return self.final_outputs

    def train(self, inputs, targets):  # 传入输入列表和期望输出列表
        inputs = np.array(inputs).T
        targets = np.array(targets).T
        outputs = self.query(inputs).T
        errors = [targets - outputs]  # 倒序储存所有层的误差,即第一个元素为输出层的误差
        for i in range(self.tier - 1):
            self.weight[-i - 1] += self.learn_rate * np.dot((errors * outputs * (1 - outputs)),
                                                            np.transpose(self.outputs_list[-i - 2]))
            outputs = self.outputs_list[-i - 2]
            errors = np.dot(self.weight[-i - 1].T, errors[i])
        # print("ereors:", errors)



    def check(self):
        print(self.nodes, 'weight', self.weight, self.tier, "out", self.outputs_list)

test = network([3, 5, 5, 5], 0.1)
test.check()
while 1:
    test.train([0.5, 0.75, 0.7], [1, 0, 0])
    print("res", test.query([0.5, 0.75, 0.7]))
test.check()