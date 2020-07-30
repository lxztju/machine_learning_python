# -*- coding:utf-8 -*-
# @time :2020/7/23
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju


'''
mnist为10分类，因为感知机算法为二分类
因此挑选其中0,1这两类的数据进行训练
'''

import numpy as np
import time
import logging


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


class Perceptron:
    def __init__(self):
        pass






    def perceptron(self, dataArr, labelArr, iters):
        '''
        构建爱呢感知机算法，其中loss function采用错误分类点的个数
        :param dataArr: 输入list格式的训练集
        :param labelArr: 输入list格式的训练集标签数据
        :param iters: 需要迭代的次数（因为数据集不保证线性可分，因此需要设置一定的迭代次数）
        :return: w， b 返回超平面的参数
        '''

        # 数据转换为numpy格式，方便进行矩阵运算
        dataMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).T
        # print(dataMat.shape)
        # print(labelMat.shape)
        # 训练数据的维度大小
        m, n = dataMat.shape
        logging.info('train data shape is:({},{})'.format(m,n))

        # 初始化为w，b
        W = np.random.randn(1, n)
        b = 0

        # 设置学习率（迭代步长）
        lr = 0.0001

        # 进行迭代训练
        for iteration in range(iters):

        # 采用sgd的方法进行权重的更新，每次选取一个错误样本更新w， b
            # 一共含有m个样本
            for i in range(m):
                #  选择某个样本
                xi = dataMat[i]
                yi = labelMat[i]
                # 如果分类正确，那么继续寻找下一个样本
                if yi * (W * xi.T + b) > 0: continue
                # 找到错误样本，更新模型参数
                W = W + lr * yi * xi
                b = b + lr * yi

            logging.info("Iteration:{} / {}".format(iteration, iters))

        return W, b



    def testPerceptron(self, dataArr, labelArr, W, b):
        '''
        测试训练得到的感知机模型的准确性
        :param dataArr:  输入list格式的测试集数据
        :param labelArr:  输入list格式的测试集数据标签
        :param w:  感知器模型超平面的法相量参数
        :param b:  感知机模型的偏置
        :return: 感知机模型在测试集的准确率
        '''

        # 数据转换为numpy格式，方便进行矩阵运算
        dataMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).T

        # 测试集的维度大小
        m ,n = dataMat.shape

        # 正确分类的样本的数目
        correct_num = 0

        # 遍历所有的测试样本，查找其中的正确分类样本个数
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]

            if (W * xi.T + b) * yi > 0:
                correct_num += 1

        return round(correct_num/m, 4)






if __name__ == '__main__':


    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='perceptron.log',
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
    logging.info('this is an info message.')

    ######################################################

    start = time.time()


    # mnist数据集的存储位置
    import os
    home = os.path.expanduser('~')
    train_path = home + '/ML/mnist/mnist_train.csv'
    test_path = home + '/ML/mnist/mnist_train.csv'

    # 读取训练与测试集
    logging.info('Loading data....')



    p = Perceptron()

    train_data_array, train_label_array = loadData(train_path)
    test_data_array, test_label_array = loadData(test_path)
    logging.info('Loading data done.')

    #训练感知机算法
    logging.info('Start training...')
    iters = 50
    w, b = p.perceptron(train_data_array, train_label_array, iters)
    logging.info('Training done.')

    # 测试感知机算法的准确率
    logging.info('Testing this model.')
    accuracy = p.testPerceptron(test_data_array, test_label_array, w, b)

    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time:{}'.format(end-start))