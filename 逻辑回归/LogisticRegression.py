# -*- coding:utf-8 -*-
# @time :2020/7/28
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju


'''
逻辑斯蒂二分类器
------------------
在计算sigmoid的exp的时候可能会出现数值较大，溢出
因此采用修正的sigmoid防止溢出
-----
修正后的sigmoid：
wx = np.dot(self.w, x)
if wx >= 0:
    probabilty = 1 /(1+ np.exp(-wx))
else:
    e = np.exp(wx)
    probabilty = e / (1 + e)
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
                labelArr.append(0)
            dataArr.append([int(num) / 255 for num in curLine[1:]])
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        # dataArr.append([int(num)/255 for num in curLine[1:]])

    # 返回data和label
    return dataArr, labelArr



class LogisticRegression:
    def __init__(self, traindataList, trainlabelList):
        for i in range(len(traindataList)):
            traindataList[i].append(1)

        self.traindataArr = np.array(traindataList)
        self.trainlabelArr = np.array(trainlabelList)
        # print(self.traindataArr.shape)
        self.w = np.zeros(self.traindataArr.shape[1])
        self.num_samples, self.num_features = self.traindataArr.shape
        self.train()

    def train(self, lr= 0.01, max_epoch= 200):
        '''
        训练得到逻辑斯蒂分类器
        :param lr: 学习率步长
        :param max_epoch: 最大的迭代次数
        :return: None，得到逻辑斯蒂分类器的权重
        '''

        for _ in range(max_epoch):
            grad = 0
            for i in range(self.num_samples):
                xi = self.traindataArr[i]
                yi = self.trainlabelArr[i]
                wx = np.dot(xi, self.w)

                ## 对sigmoid进行修正，防止溢出
                if wx >= 0:
                    grad += xi * yi -1.0/(1+np.exp(-wx)) * xi
                else:
                    e = np.exp(wx)
                    grad += xi * yi - ( e / (1+e) ) * xi
            self.w +=  lr * grad





    def predict(self, x):
        '''
        输入x，利用逻辑斯蒂回归进行预测
        :param x: 输入的x，numpy格式的array
        :return: label
        '''
        wx = np.dot(self.w, x)
        if wx >= 0:
            probabilty = 1 /(1+ np.exp(-wx))
        else:
            e = np.exp(wx)
            probabilty = e / (1 + e)
        if probabilty > 0.5:
            return 1
        else:
            return 0

    def testModel(self, testdataArr, testlabelArr):
        '''
        测试模型的准确度
        :param testdataArr: numpy array
        :param testlabelArr:  numpy array
        :return: 准确率
        '''
        # testdataArr = np.array(testdataArr)
        correct_num = 0
        for i in range(len(testdataArr)):
            # print(testdataArr[i].shape)
            if self.predict(testdataArr[i] + [1]) == testlabelArr[i]:
                correct_num += 1
        return round(correct_num / len(testdataArr), 4 )




if __name__ == '__main__':

    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='LogisticRegression.log',
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

    logging.info('Building a LogisticRegression model.')
    logisiticRegression = LogisticRegression(traindataArr, trainlabelArr)

    logging.info('Using LogisticRegression to predict one sample.')

    prediction = logisiticRegression.predict(testdataArr[0] + [1])
    logging.info('Testing processing Done,and the prediction and label are : ({},{})'.format(str(prediction), str(testlabelArr[0])))

    #测试朴决策树算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    logging.info('Testing the LogisticRegression model.')
    accuracy = logisiticRegression.testModel(testdataArr[:200], testlabelArr[:200])


    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end-start), 4))

