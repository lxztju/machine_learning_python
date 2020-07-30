# -*- coding:utf-8 -*-
# @time :2020/7/30
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com

'''
SVM的python实现
实现线性软间隔的二分类

利用SMO算法进行训练
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
            dataArr.append([int(num) / 255 for num in curLine[1:]])
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        # dataArr.append([int(num)/255 for num in curLine[1:]])

    # 返回data和label
    return dataArr, labelArr





class SVM:
    def __init__(self):
        pass


    def train(self):
        pass



    def predict(self):
        pass


    def testModel(self):
        pass





if __name__ == '__main__':
    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='SVM.log',
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
    test_path = home + '/ML/mnist/mnist_train.csv'

    # 读取训练与测试集
    logging.info('Loading data....')

    traindataArr, trainlabelArr = loadData(train_path)
    testdataArr, testlabelArr = loadData(test_path)
    logging.info('Loading data done.')

    num_classes = 10
    num_features = 28 * 28
    svm = SVM(num_classes, num_features, traindataArr, trainlabelArr)

    # 测试朴素贝页斯算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    accuracy = svm.testModel(testdataArr[:200], testlabelArr[:200])

    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end - start), 4))





















