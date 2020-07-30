# -*- coding:utf-8 -*-
# @time :2020/7/28
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

'''
实现CART分类树
这个算法与ID3和C4.5算法的主要不同在于采用Gini指数来选择特征
-----------
未剪枝
'''

import numpy as np
import logging
import time
import copy

def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    # 存放数据及标记的list
    dataArr = []
    labelArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')

        labelArr.append(int(curLine[0]))
        # 进行二值化处理，将大于128的标记为1， 小于128的标记为0
        dataArr.append([int(int(num)>128) for num in curLine[1:]])
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)

    # 返回data和label
    return dataArr, labelArr




class CARTDecisionTree:
    def __init__(self, traindataList, trainlabelList):
        '''
        初始化决策树类
        :param traindataList:  训练数据集的list形式
        :param trainlabelList:  训练数据集的label的list形式
        '''
        self.traindataList = traindataList
        self.trainlabelList = trainlabelList
        self.traindataArr = np.array(self.traindataList)
        self.trainlabelArr = np.array(self.trainlabelList)


        self.tree = self.build_CARTtree(self.traindataArr, self.trainlabelArr)
        print(self.tree)
    def calculate_Gini(self, trainlabelArr):
        '''
        计算数据集D的Gini指数
        # :param traindataArr: 训练数据集 numpy格式
        :param trainlabelArr:  训练数据集label numpy格式
        :return:  返回Gini指数
        '''
        D = trainlabelArr.size
        labels = set([label for label in trainlabelArr])
        Gini = 1
        for label in labels:
            Ck = trainlabelArr[trainlabelArr==label].size
            Gini -= ( Ck /D) ** 2
        return Gini



    def calculate_Gini_feature(self, trainfeatureArr, trainlabelArr, a):
        '''
        计算数据集针对特征A的取值为a时的Gini指数
        :param trainfeatureArr: 切分后的训练数据集某一个特征列 numpy格式
        :param trainlabelArr: 训练数据集label numpy格式
        :param a: 特征A的某一个取值
        :return:  返回基尼指数
        '''
        D1 = trainfeatureArr[trainfeatureArr == a].size
        D = trainfeatureArr.size
        D2 = D - D1
        d1 = trainlabelArr[trainfeatureArr == a]
        d2 = trainlabelArr[trainfeatureArr != a]

        Gini_D_A = abs(D1/D) * self.calculate_Gini(d1) + abs(D2/D) * self.calculate_Gini( d2 )

        return Gini_D_A

    def calculate_min_Gini(self, traindataArr, trainlabelArr):
        '''
        计算最小的Gini指数与对应的特征
        :param traindataArr:  训练数据集 numpy格式
        :param trainlabelArr:  训练数据集的label numpy格式
        :return:  返回最小的Gini指数与对应的特征
        '''
        num_features = traindataArr.shape[1]
        min_Gini = float('inf')
        feature = -1
        v = -1
        for i in range(num_features):
            trainfeatureArr = traindataArr[:, i]
            values = set([value for value in trainfeatureArr])
            for value in values:
                gini = self.calculate_Gini_feature(trainfeatureArr, trainlabelArr, value)
                if gini < min_Gini:
                    min_Gini = gini
                    feature = i
                    v = value

        return feature, v, min_Gini




    def updateDataSetleft(self, traindataArr, trainlabelArr, A, a):
        '''
        在构建决策树的过程中，需要实时更新决策树的数据集
        :param traindataArr: 待更新的数据集，numpy格式
        :param trainlabelArr:  待更新的数据集label， numpy格式
        :param A:  需要删除的特征
        :param a: 对于需要删除的特征A，如果其取值为a，那说明这个样本需要保留（解释一下，例如对于是否有工作这个特征，a为有工作
                    那么所有有工作的样本需要保留。
        :return: 返回新的数据集及标签，numpy格式
        '''
        newdataArr = np.delete(traindataArr[traindataArr[:,A] == a], A, axis=1)
        newlabelArr = trainlabelArr[traindataArr[:,A] == a]
        return newdataArr, newlabelArr


    def updateDataSetright(self, traindataArr, trainlabelArr, A, a):
        '''
        在构建决策树的过程中，需要实时更新决策树的数据集
        :param traindataArr: 待更新的数据集，numpy格式
        :param trainlabelArr:  待更新的数据集label， numpy格式
        :param A:  需要删除的特征
        :param a: 对于需要删除的特征A，如果其取值为a，那说明这个样本需要保留（解释一下，例如对于是否有工作这个特征，a为有工作
                    那么所有有工作的样本需要保留。
        :return: 返回新的数据集及标签，numpy格式
        '''
        newdataArr = np.delete(traindataArr[traindataArr[:,A] != a], A, axis=1)
        newlabelArr = trainlabelArr[traindataArr[:,A] != a]
        return newdataArr, newlabelArr


    def majorClass(self, trainlabelArr):
        '''
        在label中找到数量最多的类别
        :param trainlabelArr: 训练数据集的label， numpy格式的
        :return:  返回最大的类别
        '''
        label = list(trainlabelArr)
        return max(label, key=label.count)


    def build_CARTtree(self, traindataArr, trainlabelArr):
        '''
        在数据集上递归构建决策树
        :param traindataArr: 当前节点为根节点对应的数据集 numpy
        :param trainlabelArr:  当前节点为根节点对应的数据集label numpy
        :return: 返回节点的值
        '''
        # 信息增益的阈值
        epsilon = 0.1

        node_thresh = 5


        # logging.info('Starting create a new Node. Now there are {} samples'.format(trainlabelArr.size))

        # 判断数据集此时的类别，如果只有一类，就范会对应的类别
        classDict = set(trainlabelArr)
        # print(classDict)
        if len(classDict) == 1:
            return int(classDict.pop())
        # print(traindataArr.shape)
        # 判断数据集此时的的特征数目，如果没有特征集，那就说明没有特征进行分割，就放会这些样本中数目最多的类别
        if len(traindataArr.shape) == 1:
            return self.majorClass(trainlabelArr)
        # 计算最大增益及其对应的特征
        Ag, a, Gini = self.calculate_min_Gini(traindataArr, trainlabelArr)
        # print(Ag, Gini)
        # 如果最大的信息增益小于设定的阈值，就直接返回数目最多的类，不必要进行分割
        if Gini < epsilon:
            return self.majorClass(trainlabelArr)

        if trainlabelArr.size < node_thresh:
            return self.majorClass(trainlabelArr)

        tree = {Ag:{}}
        # 递归构建决策树


        newdataArrleft, newlabelArrleft = self.updateDataSetleft(traindataArr, trainlabelArr, Ag, a)
        newdataArrright, newlabelArrright = self.updateDataSetright(traindataArr, trainlabelArr, Ag, a)
        # print(newlabelArrleft.size, newlabelArrright.size, trainlabelArr.size)
        if newlabelArrleft.size > 0:
            tree[Ag][a] = {'left': self.build_CARTtree(newdataArrleft, newlabelArrleft)}
        if newlabelArrright.size > 0:
            tree[Ag][a]['right'] = self.build_CARTtree(newdataArrright, newlabelArrright)

        # print(tree)
        return tree

    def predict(self, testdataList):
        '''
        使用构建完成的决策树来预测对应的测试数据
        :param testdataList: 输入的行测试数据，list格式
        :return:  返回类别
        '''
        tree = copy.deepcopy(self.tree)
        # print(tree)
        while True:
            if type(tree).__name__ != 'dict':
                return tree
            # print(tree.items())
            (key, value), = tree.items()

            if type(tree[key]).__name__ == 'dict':
                dataval = testdataList[key]

                del testdataList[key]

                k = list(value.keys())
                if dataval not in k:
                    tree = value[k[0]]['right']
                else:
                    tree = value[dataval]['left']

                if type(tree).__name__ != 'dict':
                    return tree

            else:
                return value



    def testModel(self, testdataList, testlabelList):
        '''
        测试决策树模型的准确率
        :param testdataList: 输入测试集的数据
        :param testlabelList: 输入测试集数据的label
        :return: 准确率accuracy
        '''
        #
        correct_num = 0

        for i in range(len(testdataList)):
            prediction = self.predict(testdataList[i])
            if prediction == testlabelList[i]:
                correct_num += 1

        return round(correct_num/len(testlabelList), 4)






if __name__ == '__main__':

    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='CART_decision_tree.log',
                        filemode='w')  # filemode默认为a，追加信息到日志文件，指定为‘w'，重新写入 文件，之前的文件信息丢失
    # 定义一个handler来将信息输出到控制台，StreamHandler与FileHandler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 设置在控制台输出格式[-
    formatter = logging.Formatter('%(asctime)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    #  将handler加入到根记录器
    logging.getLogger('').addHandler(console)

    # 根记录器输出信息
    logging.info('This is an info message.')


    start = time.time()


    # mnist数据集的存储位置
    import os
    home = os.path.expanduser('~')
    # train_path = home + '/ML/mnist/mnist_train.csv'
    # test_path = home + '/ML/mnist/mnist_test.csv'
    train_path = home + '/ML/mnist/mnist_train_samples.csv'
    test_path = home + '/ML/mnist/mnist_test_samples.csv'

    # 读取训练与测试集
    logging.info('Loading data....')

    traindataArr, trainlabelArr =loadData(train_path)
    testdataArr, testlabelArr = loadData(test_path)
    logging.info('Loading data done.')

    logging.info('Building a decision tree.')
    CART = CARTDecisionTree(traindataArr, trainlabelArr)

    logging.info('Using decision tree to predict one sample.')

    prediction = CART.predict(testdataArr[0])
    logging.info('Testing processing Done,and the prediction and label are : ({},{})'.format(str(prediction), str(testlabelArr[0])))

    # 测试朴决策树算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    logging.info('Testing /the decision model.')
    accuracy = CART.testModel(testdataArr[:200], testlabelArr[:200])


    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end-start), 4))

