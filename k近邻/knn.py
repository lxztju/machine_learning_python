# -*- coding:utf-8 -*-
# @time :2020/7/24
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju


'''
实现一个KNN算法，并实现两种进行最近邻搜索的方法，
线性搜索最近邻
---------------
距离的度量采用欧式距离与曼哈顿距离计算
'''

import numpy as np
import time
import logging




class Knn:
    def __init__(self, k, num_classes, dist_method):
        self.k = k
        self.num_classes = num_classes
        self.dist_method = dist_method



    def loadData(self, fileName):
        '''
        加载Mnist数据集
        :param fileName:要加载的数据集路径
        :return: list形式的数据集及标记
        '''
        # 存放数据及标记的list
        dataArr = []; labelArr = []
        # 打开文件
        fr = open(fileName, 'r')
        # 将文件按行读取
        for line in fr.readlines():
            # 对每一行数据按切割福','进行切割，返回字段列表
            curLine = line.strip().split(',')


            labelArr.append(int(curLine[0]))
            dataArr.append([int(num) / 255 for num in curLine[1:]])
            #存放标记
            #[int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素（标记）外将所有元素转换成int类型
            #[int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)

        #返回data和label
        return dataArr, labelArr




    def calculate_distance(self, x1, x2):
        '''
        计算两个向量之间的距离
        :param x1: 第一个向量，numpy格式的列向量
        :param x2: 第二个向量，numpy格式的列向量
        :param method: 值为'l2, l1'，l2为欧式距离， l1为曼哈顿聚类
        :return: 返回距离度量值，标量值
        '''
        if self.dist_method == 'l2':
            return np.sqrt(np.sum(np.square(x1 - x2)))
        else:
            return np.sum(np.abs(x1 - x2))





    def linear_get_k_cloest(self, dataMat, labelMat, x):
        '''
        构建爱呢感知机算法，其中loss function采用错误分类点的个数
        :param dataMat: 输入numpy格式的训练集
        :param labelMat: 输入numpy格式的训练集标签数据
        :param x: 待查验的向量
        :return: label , knn预测的label值
        '''

        # 训练数据的维度大小
        m, n = dataMat.shape

        ##线性遍历每个节点，分别记录各个节点的距离，然后找到最近邻的k个节点
        dists = [0] * m  # 记录每个节点与待查节点的距离
        for i in range(m):
            xi = dataMat[i]

            dist = self.calculate_distance(xi, x)
            dists[i] = dist

        # 得到待测点与所有点的距离值，然后将所有的距离值排序，找到最近的k距离值的索引
        # argsort返回从小到大排序的元素的索引
        topk_index = np.argsort(np.array(dists))[:self.k]
        # print(type(topk_index), topk_index)
        # labelList表示每个类别的近邻样本的数目
        labelList = [0] * self.num_classes
        for index in topk_index:
            labelList[int(labelMat[index])] += 1
        # 返回识别后的类别
        return labelList.index(max(labelList))




    def modelTest(self, traindataArr, trainlabelArr, testdataArr, testlabelArr):
        '''
        测试knn模型的准确率
        :param traindataArr:  训练数据的list格式
        :param trainLabelArr:  测试数据label的list格式
        :param testdataArr:  测试数据的list个格式存储
        :param testlabelArr:  测试数据label的list格式
        :return:
        '''

        # 数据转换为numpy格式，方便进行矩阵运算
        traindataMat = np.mat(traindataArr)
        trainlabelMat = np.mat(trainlabelArr).T
        testdataMat = np.mat(testdataArr)
        testlabelMat = np.mat(testlabelArr).T

        # 测试集的维度大小
        m ,n = testdataMat.shape
        m1, n1 = traindataMat.shape
        logging.info('test data shape is:({},{})'.format(m,n))
        logging.info('train data shape is:({},{})'.format(m1,n1))


        # 正确分类的样本的数目
        correct_num = 0

        # 遍历所有的测试样本，查找其中的正确分类样本个数
        for i in range(m):
            xi = testdataMat[i]
            yi = testlabelMat[i]
            if i % 50 == 0:
                logging.info('Testing data:({}/{}), and correct_num:{}'.format(i, m, correct_num))
            # 统计分类正确的元素点的个数
            if self.linear_get_k_cloest(traindataMat, trainlabelMat, xi) == yi:
                correct_num += 1

        return round(correct_num/m, 4)






if __name__ == '__main__':

    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='knn.log',
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

    topk = 20
    num_classes = 10
    dist_method = 'l2'
    knn = Knn(topk, num_classes, dist_method)

    # 读取训练与测试集
    logging.info('Loading data....')

    traindataArr, trainlabelArr = knn.loadData(train_path)
    testdataArr, testlabelArr = knn.loadData(test_path)
    logging.info('Loading data done.')

    #测试knn算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    accuracy = knn.modelTest(traindataArr, trainlabelArr, testdataArr[:200], testlabelArr[:200])


    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end-start), 4))
