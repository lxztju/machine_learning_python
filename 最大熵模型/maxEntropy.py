# -*- coding:utf-8 -*-
# @time :2020/7/29
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju


''''
构建最大熵模型，并采用mnist数据集进行训练策测试

--------------------------
没搞懂
'''
import numpy as np
import logging
import time


def loadData(filename):
    '''
    加载mnist数据集
    :param filename: 待加载的数据集路径
    :return: 返回加载后的数据集list
    '''
    dataList = []
    labelList =  []

    f = open(filename, 'r')

    for line in f.readlines():

        curdata = line.strip().split(',')

        labelList.append(int(curdata[0]))

        dataList.append([int(int(value)>128) for value in curdata[1:]])

    return dataList, labelList




class MaxEntropy:
    def __init__(self, traindataList, trainlabelList):

        self.traindataArr = np.array(traindataList)
        self.trainlabelArr = np.array(trainlabelList)



if __name__ == '__main__':
    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='maxEntropy.log',
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

    traindataArr, trainlabelArr = loadData(train_path)
    testdataArr, testlabelArr = loadData(test_path)
    logging.info('Loading data done.')

    logging.info('Building a LogisticRegression model.')
    maxEntropy = MaxEntropy(traindataArr, trainlabelArr)

    logging.info('Using LogisticRegression to predict one sample.')

    prediction = maxEntropy.predict(testdataArr[0] + [1])
    logging.info('Testing processing Done,and the prediction and label are : ({},{})'.format(str(prediction),
                                                                                             str(testlabelArr[
                                                                                                     0])))

    # 测试朴决策树算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    logging.info('Testing the LogisticRegression model.')
    accuracy = maxEntropy.testModel(testdataArr[:200], testlabelArr[:200])

    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end - start), 4))