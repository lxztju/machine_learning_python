# -*- coding:utf-8 -*-
# @time :2020/7/25
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

'''
构建决策树
ID3算法实现决策树（不剪枝）
ID3采用信息增益作为特征选择的标准
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




class ID3DecisionTree:
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


        self.tree = self.build_ID3tree(self.traindataArr, self.trainlabelArr)


    def calculate_empirical_entropy(self, trainLabelArr):
        '''
        计算训练数据集的经验熵，公式参考李航老师统计学习方法
        :param trainLabelArr: numpy格式的label
        :return: 返回训练集的经验熵
        '''
        # 初始化经验熵为0
        H_D = 0
        # 这里为什么不采用self.num_classes直接调用，我刚开始也是这么写的
        # 后来发现如果在后期的计算中，某个类别不出现，那么log0会出现错误（参考README.md参考链接中大佬的利用set的实现）
        labels = set([label for label in trainLabelArr])
        for label in labels:

            # 根据公式需要计算每个类别的数目
            num = trainLabelArr[trainLabelArr==label].size
            # 计算每个类别占据数目占据整个数据集的比例
            p = num / trainLabelArr.size
            # 计算经验熵
            H_D += -1 *(p) * np.log2(p)

        return H_D




    def calculate_empirical_conditional_entropy(self, trainfeatureArr, trainlabelarr):
        '''
        计算经验条件熵
        :param trainfeatureArr: numpy格式的从数据集中抽离出某一个特征列
        :param trainlabelabelArr: numpy格式的label
        :return: 经验条件熵
        '''

        # 经验熵是对每个特征进行计算，因此应该返回一个列表，对于每个特征都进行计算分析
        # 桶计算经验熵时一样，采用set来选取特针的不同取值
        features = set([feature for feature in trainfeatureArr])
        H_D_A = 0
        for feature in features:
            # 计算取不同值时所包含的样本的数目
            Di = trainfeatureArr[trainfeatureArr == feature].size
            Di_D = Di / trainfeatureArr.size

            # 计算对于选取的特征取feature值时的条件熵

            H_D_A += Di_D * self.calculate_empirical_entropy(trainlabelarr[trainfeatureArr == feature])

        return H_D_A


    def calculate_information_gain(self, traindataArr, trainlabelArr):
        '''
        :param traindataArr: 当前数据集的数组，numpy格式，因为每次在构建决策树机型分支的过程中，随着决策树层数的加深当前数据集会比越变越小
        :param trainlabelArr: 当前数据集的label数组，numpy格式
        计算最大的信息增益
        :return: 最大的信息增益及其对应的特征。
        '''
        # 获取当前数据集的特征数目
        num_features = traindataArr.shape[1]
        max_feature, max_G_D_A = 0, 0
        # 计算当前数据集的经验熵
        H_D = self.calculate_empirical_entropy(trainlabelArr)
        # 计算每个特征的经验条件熵
        for i in range(num_features):
            trainfeatureArr = traindataArr[:,i]
            H_D_i = self.calculate_empirical_conditional_entropy(trainfeatureArr, trainlabelArr)
            G_D_A = H_D - H_D_i
            if G_D_A > max_G_D_A:
                max_G_D_A = G_D_A
                max_feature  = i
        # 返回最大的信息增益，及其特征
        return max_feature, max_G_D_A


    def updateDataSet(self, traindataArr, trainlabelArr, A, a):
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


    def majorClass(self, trainlabelArr):
        '''
        在label中找到数量最多的类别
        :param trainlabelArr: 训练数据集的label， numpy格式的
        :return:  返回最大的类别
        '''
        label = list(trainlabelArr)
        return max(label, key=label.count)


    def build_ID3tree(self, traindataArr, trainlabelArr):
        '''
        在数据集上递归构建决策树
        :param traindataArr: 当前节点为根节点对应的数据集 numpy
        :param trainlabelArr:  当前节点为根节点对应的数据集label numpy
        :return: 返回节点的值
        '''
        # 信息增益的阈值
        epsilon = 0.1


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
        Ag, G_D_Ag = self.calculate_information_gain(traindataArr, trainlabelArr)
        # print(Ag, G_D_Ag)
        # 如果最大的信息增益小于设定的阈值，就直接返回数目最多的类，不必要进行分割
        if G_D_Ag < epsilon:
            return self.majorClass(trainlabelArr)

        tree = {Ag:{}}
        # 递归构建决策树
        features = set(feature for feature in traindataArr[:, Ag])
        for feature in features:
            a = int(feature)
            newdataArr, newlabelArr = self.updateDataSet(traindataArr, trainlabelArr, Ag, a)

            tree[Ag][a] = self.build_ID3tree(newdataArr, newlabelArr)
        # print(tree)
        return tree

    def predict(self, testdataList):
        '''
        使用构建完成的决策树来预测对应的测试数据
        :param testdataList: 输入的行测试数据，list格式
        :return:  返回类别
        '''
        tree = copy.deepcopy(self.tree)
        while True:
            if type(tree).__name__ != 'dict':
                return tree
            # print(tree.items())
            (key, value), = tree.items()

            if type(tree[key]).__name__ == 'dict':
                dataval = testdataList[key]

                del testdataList[key]
                tree = value[dataval]

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
                        filename='ID3_decision_tree.log',
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
    train_path = home + '/ML/mnist/mnist_train.csv'
    test_path = home + '/ML/mnist/mnist_test.csv'
    # train_path = home + '/ML/mnist/mnist_train_samples.csv'
    # test_path = home + '/ML/mnist/mnist_test_samples.csv'

    # 读取训练与测试集
    logging.info('Loading data....')

    traindataArr, trainlabelArr =loadData(train_path)
    testdataArr, testlabelArr = loadData(test_path)
    logging.info('Loading data done.')

    logging.info('Building a decision tree.')
    ID3 = ID3DecisionTree(traindataArr, trainlabelArr)

    logging.info('Using decision tree to predict one sample.')

    prediction = ID3.predict(testdataArr[0])
    logging.info('Testing processing Done,and the prediction and label are : ({},{})'.format(str(prediction), str(testlabelArr[0])))

    #测试朴决策树算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    logging.info('Testing the decision model.')
    accuracy = ID3.testModel(testdataArr[:200], testlabelArr[:200])


    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end-start), 4))

