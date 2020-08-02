# -*- coding:utf-8 -*-
# @time :2020/8/1
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com

'''
实现AdaBoost的提升方法

利用mnist数据集其中的两类进行分类

----------------
利用统计学习方法中例题8.1.3,利用阈值分割（单层决策树）作为基分类器


'''

import numpy as np
import logging
import time


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

        # Mnsit有0-9是个标记，由于是二分类任务，所以仅仅挑选其中的0和1两类作为正负类进行分类
        # if int(curLine[0]) != 0 or int(curLine[0]) !=1: continue
        if int(curLine[0]) == 0 or int(curLine[0]) == 1:
            if int(curLine[0]) == 0:
                labelArr.append(1)
            else:
                labelArr.append(-1)
            dataArr.append([int(int(num) > 128) for num in curLine[1:]])

    # 返回data和label
    return dataArr, labelArr



class SingleTree:
    def __init__(self, traindataList, trainlabelList):
        '''
        构建单层的决策树作为AdaBoost的基分类器
        :param traindataList:  输入的数据集的list格式
        :param trainlabelList: 输入训练集的label的list格式
        :param D: 训练数据集的权重
        '''
        self.traindataArr = np.array(traindataList)
        self.trainlabelArr = np.array(trainlabelList)
        self.m, self.n = self.traindataArr.shape
        self.D = [1/ self.m] * self.m  # 初始化数据集权重为均匀分布


    def calcError(self, prediction, trainlabelArr, D):
        '''
        计算在训练数据集上的分类误差率
        :param prediction:  决策树预测出的prediction，与trainlabelArr长度相同
        :param trainlabelArr:  ground truth
        :param D:  训练数据集的权重
        :return: 返回训练误差率
        '''
        # 初始化error
        error = 0

        for i in range(trainlabelArr.size):
            if prediction[i] != trainlabelArr[i]:
                error += D[i]
        return error


    def singleTree(self):
        '''
        构建单层决策树，作为基分类器
        :return:
        '''
        # 利用字典构建一棵树
        # print(self.D)
        tree = {}
        # 切分点，由于数据集读取的过程中，每个特征的取值均为0 和 1,因此选择三个切分点，第一个小于0,第二个0,1之间，第三个大于1
        divides = [-0.5, 0.5, 1.5]
        # 指定规则，对于某个特征，less为小于切分点阈值的为1,大于的为-1
        #                     Over为大于切分点阈值的为-1, 小于的为1
        rules = ['Less', 'Over']
        # 最大的误差值为1,因此初始化为1
        min_error = 1
        # 遍历每个特征，找寻能够使得误差最小值的切分店，与切分规则还有特征值
        for i in range(self.n):
            for divide in divides:

                for rule in rules:
                    #初始化预测的结果为predicition
                    prediction = np.ones(self.m)
                    if rule == 'Less':
                        # 当切分规则为Less时，大于切分点的样本置为-1,因为一开始一开始初始化为1，因此预测为1的可不进行赋值处理
                        prediction[self.traindataArr[:,i] >divide] = -1
                    else:
                        # 当切分点为Over时，小于切分店的样本置为-1
                        prediction[self.traindataArr[:, i] <= divide] = -1
                    # 对于给定的特征、切分点、切分规则，计算相对应的错误率
                    error = self.calcError(prediction, self.trainlabelArr, self.D)
                    # 找到最小的错误率来构建树
                    if error < min_error:
                        # print(prediction, self.traindataArr[:, i], trainlabelList)
                        tree['error'] = error
                        tree['rule'] = rule
                        tree['divide'] = divide
                        tree['feature'] = i
                        tree['Gx'] = prediction
                        min_error = error
        # print(tree, error)
        return tree


class Adaboost(SingleTree):
    def __init__(self, traindataList, trainlabelList, treeNum = 50):
        super().__init__(traindataList, trainlabelList)

        self.treeNum = treeNum

        self.trees = self.BoostingTree()



    def BoostingTree(self):
        '''
        构建Adaboost
        :return: 返回构建完成的Adaboost模型
        '''
        # 初始化树的列表，每个元素代表一棵树，从前到后一层层
        tree = []
        # 最终的预测值列表，每个元素表示对于每个样本的预测值
        finalPrediction = np.zeros(self.trainlabelArr.size)
        #迭代生成treeNum层的树
        for i in range(self.treeNum):
            # 构建单层的树
            curTree = self.singleTree()
            # 根据公式8.2,计算alpha
            alpha = 1/2 * np.log((1-curTree['error']) / curTree['error'])
            # 保留这一层树的预测值，用于后边权重值的计算
            Gx = curTree['Gx']

            # 计算数据集的权重
            # 式子8.4的分子部分，是一个向量，在array中 *与np.multiply表示元素对应相乘
            # np.dot()是向量点乘
            w = self.D * ( np.exp( -1 * alpha * self.trainlabelArr * Gx))
            # 训练集的权重分布
            self.D = w / sum(w)
            curTree['alpha'] = alpha
            # print(curTree)

            tree.append(curTree)

            #################################
            # 计算boosting的效果，提前中止
            finalPrediction += alpha * Gx
            # print(finalPrediction, self.trainlabelArr, alpha)
            correct_num = sum(np.sign(finalPrediction) == self.trainlabelArr)
            # print(correct_num, finalPrediction, self.trainlabelArr)
            accuracy = correct_num / self.trainlabelArr.size
            logging.info("The {}th Tree, The train data's accuracy is:{}".format(i, accuracy))
            # 如果在训练集上转却率已经达到1,提前中止
            if accuracy == 1:
                break
        return tree

    def predict(self, x, div, rule, feature):
        '''
        对于单个样本，来计算基分类器的输出结果
        :param x: 输入样本
        :param div: 拆分点的阈值
        :param rule: 拆分规则， Less 或者 Over
        :param feature: 对应操作的特征
        :return: 返回预测的label
        '''

        if rule == 'Less':
            L, H = 1, -1
        else:
            L, H = -1, 1

        if x[feature] > div:
            return H
        else:
            return L



    def testModel(self, testdataList, testlabelList):
        '''
        预测Adaboost模型的准确率
        :param testdataList:  输入的测试集的list格式
        :param testlabelList:  测试集的label
        :return: 返回准确率
        '''
        correct_num = 0

        for i in range(len(testdataList)):
            result = 0

            for curTree in self.trees:

                div = curTree['divide']
                feature = curTree['feature']
                rule = curTree['rule']
                alpha = curTree['alpha']
                result += alpha * self.predict(testdataList[i], div, rule, feature)

            if np.sign(result) == testlabelList[i]:
                correct_num += 1

        return round((correct_num /len(testlabelList)* 100), 4)



if __name__ == '__main__':

    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='AdaBoost.log',
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

    traindataList, trainlabelList = loadData(train_path)
    testdataList, testlabelList = loadData(test_path)
    logging.info('Loading data done.')
    # print(trainlabelList[:100])
    logging.info('Training the AdaBoost model....')

    adaboost = Adaboost(traindataList[:1000], trainlabelList[:1000])


    # logging.info('Predicting one sample ....')
    # prediction = adaboost.predict([testdataList[0]], [testlabelList[0]])
    # logging.info('The prediction and the ground truth is : ({}, {})'.format(prediction, testlabelList[0]))

    # 测试Adaboost算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    accuracy = adaboost.testModel(testdataList, testlabelList)

    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end - start), 4))