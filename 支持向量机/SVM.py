# -*- coding:utf-8 -*-
# @time :2020/7/30
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com

'''
SVM的python实现
实现软间隔与核函数的非线性SVM二分类器

利用SMO算法进行训练
'''

import numpy as np
import logging
import time
import random
import math

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
    def __init__(self, traindataList, trainlabelList, sigma = 10, C = 200, toler = 0.001):
        '''
        SVM类的参数初始化
        :param traindataList: 训练数据集的LIst格式
        :param trainlabelList:  训练数据集label的List格式
        :param sigma: 高斯核的参数
        :param C:  软间隔的惩罚参数
        :param toler: 松弛变量
        '''
        self.traindataArr = np.array(traindataList)   # 训练数据集转换为array格式
        self.trainlabelArr = np.array(trainlabelList).T  # 训练数据集的label转换为array格式，进行转置变成列向量
        self.m, self.n = self.traindataArr.shape   # m为训练集的样本个数， n为特征的个数

        self.sigma = sigma  # 高斯核中的参数
        self.C = C   #软间隔的惩罚参数
        self.toler = toler # 松弛变量
        self.b = 0   # SVM中的偏置项
        self.alpha = [1] * self.traindataArr.shape[0]  # SVM对偶问题中的alpha
        self.kernel = self.calcKernel()  # 核函数矩阵
        self.E = [self.calc_Ei(i) for i in range(self.m)]    #SMO运算过程中的Ei
        # print(self.E)
        self.supportVecIndex = []   # 保存支持向量的索引




    def calcKernel(self):
        '''
        计算核函数矩阵，采用高斯核
        :return: 高斯核矩阵
        '''

        # 高斯核矩阵的大小为m×m
        K = [[0] * self.m for _ in range(self.m)]

        # 遍历Xi， 这个相当于核函数方程中的x
        for i in range(self.m):

            if i % 100 == 0:
                logging.info('Construct The Gaussian Kernel: ({}/{}).'.format(i, self.m))

            Xi = self.traindataArr[i]
            #遍历Xj，相当于公式中的Z
            for j in range(self.m):
                Xj = self.traindataArr[j]
                # 计算||xi-xj||^2
                diff = np.dot((Xi - Xj), (Xi - Xj).T)
                # nisan高斯核参数矩阵
                K[i][j] = np.exp((-1/2) * (diff/(self.sigma ** 2 )))

        # 返回高斯核
        return K



    def calc_gxi(self, i):
        '''
        根据7.104的公式计算g(xi)
        :param i:  x的下标
        :return: 返回g(xi)的值
        '''
        gxi = 0
        for j in range(len(self.alpha)):
            gxi += self.alpha[j] * self.trainlabelArr[j] * self.kernel[i][j]

        return gxi + self.b



    def calc_Ei(self, i):
        '''
        计算公式7.104，计算Ei
        :param i: 下标
        :return: Ei
        '''
        gxi = self.calc_gxi(i)
        return gxi - self.trainlabelArr[i]



    def isSatisfyKKT(self, i):
        '''
        判断第i个alpha是否满足KKT条件， 因为在SMO算法中
        第一个alpha的选取采用最不符合KKT条件的哪一个
        :param i: alpha的下标i
        :return: True or False
        '''
        gxi = self.calc_gxi(i)
        yi = self.trainlabelArr[i]
        multiply = gxi * yi
        alpha_i = self.alpha[i]

        #  书中采用的是alpha等于0，但是可以进行松弛操作
        # if alpha_i == 0:
        if (abs(self.alpha[i]) < self.toler) and (multiply >= 1):
             return True
        # 哦嗯样均采用松弛之后的
        # if alpha_i == self.C:
        if abs(self.alpha[i] - self.C) < self.toler and (multiply <= 1):
            return True

        #if 0 < alpha_i < self.C:
        if (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) and (multiply < 1 + self.toler):
            return True

        return False



    def getAlpha(self):
        '''
        SMO算法的2个变量
        :return: 返回E1, E2, i, j
        '''
        # 首先遍历所有支持向量点，如果全部满足KKt条件，然后再去所有的数据集中查找
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self.isSatisfyKKT(i):
                continue
            E1 = self.E[i]

            # 如果E1为正，你那么找到最小的E作为E2保证|E1-E2|最大
            E = {k:v for v, k in enumerate(self.E)}
            E_ = sorted(E.items(), key=lambda item: item[0])

            if E1 >= 0:
                j = E_[0][1]
                # 如果找到的j与i相同，此时i代表的值最小，因此选择下一个值，如果不进行处理，使得i， j相同，那么后边会出现错误
                if j == i:
                    j = E_[1][1]
                # j = min(range(self.m), key = lambda x:self.E[x])
            # 如果E1为负，你那么找到最大的E作为E2保证|E1-E2|最大
            else:
                j = E_[-1][0]
                if j == i:
                    j = E[-2][1]
                # j = max(range(self.m), key = lambda x:self.E[x])
            # print(type(i), type(j))
            j = int(j)
            E2 = self.E[j]
        return E1, E2, i, j


    def train(self, iter = 100):
        '''
        训练SVM分类器
        :param iter: 最大的迭代次数
        :return:  无返回值，训练SVM
        '''
        iterStep = 0   # 迭代的次数，超过迭代次数依然没有收敛，则强制停止
        parameterChanged = 1 # 参数是否发生更改的标志，如果发生更改，那么这个值为1,如果不更改，说明算法已经收敛

        # 迭代训练SVM
        while iterStep < iter and parameterChanged > 0:
            logging.info('Iter:{}/{}'.format(iterStep, iter))

            iterStep += 1
            # 初始化参数变化值为0,如果参数改变，说明训练过程正在进行，那么parameterChanged置一
            parameterChanged = 0

            # 利用SMO更新的两个变量
            E1, E2, i, j = self.getAlpha()

            y1 = self.trainlabelArr[i]
            y2 = self.trainlabelArr[j]

            alpha1Old = self.alpha[i]
            alpha2Old = self.alpha[j]

            # 计算边界
            if y1 == y2:
                L = max(0, alpha2Old+alpha1Old-self.C)
                H = min(self.C, alpha2Old + alpha1Old)
            else:
                L = max(0, alpha2Old-alpha1Old)
                H = min(self.C, self.C+alpha2Old+alpha1Old)

            if L == H:
                continue
            # print(L, H, alpha1Old, alpha2Old)
            k11 = self.kernel[i][i]
            k22 = self.kernel[j][j]
            k12 = self.kernel[i][j]
            k21 = self.kernel[j][i]

            eta = (k11 + k22 - 2*k12)

            # 如果eta为0,在后边的分母中会报错
            if eta <= 0:
                continue

            alpha2NewUnc = alpha2Old + y2 * (E1-E2)/ eta
            # print(E1, E2, eta, alpha2Old, alpha2NewUnc)
            if alpha2NewUnc <L:
                alpha2New = L
            elif alpha2NewUnc > H:
                alpha2New = H
            else:
                alpha2New = alpha2NewUnc
            # print(alpha2New, alpha2Old)
            alpha1New = alpha1Old + y1 * y2 * (alpha2Old - alpha2New)

            b1New = -1 * E1 - y1 * k11 * (alpha1New - alpha1Old) \
                    - y2 * k21*(alpha2NewUnc - alpha2Old) + self.b

            b2New = -1 * E2 - y1 * k12 * (alpha1New - alpha1Old) \
                    - y2 * k22 * (alpha2New - alpha2Old) + self.b

            # 依据α1和α2的值范围确定新b
            if (alpha1New > 0) and (alpha1New < self.C):
                bNew = b1New
            elif (alpha2New > 0) and (alpha2New < self.C):
                bNew = b2New
            else:
                bNew = (b1New + b2New) / 2

            self.alpha[i] = alpha1New
            self.alpha[j] = alpha2New
            self.b = bNew

            self.E[i] = self.calc_Ei(i)
            self.E[j] = self.calc_Ei(j)
            # parameterChanged = 1
            # print(abs(alpha2New - alpha2Old))
            # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
            # 反之则自增1
            if abs(alpha2New - alpha2Old) >= 0.00001:
                parameterChanged = 1
            # break
        #全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            #如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                #将支持向量的索引保存起来
                self.supportVecIndex.append(i)

        logging.info('Training process is Done !!!!')


    def predict(self, x):
        '''
        输入单个样本计算输出
        :param x:  输入的待预测的样本, list格式
        :return: 返回预测的label值
        '''
        x = np.array(x)

        result = 0
        ## 只有支持向量起作用
        for i in self.supportVecIndex:
            x1 = self.traindataArr[i]
            diff = np.dot((x1 - x), (x1 - x).T)
            k = np.exp((-1/2) * diff /(self.sigma ** 2))
            result += self.alpha[i] * self.trainlabelArr[i] * k
        result += self.b
        return np.sign(result)


    def testModel(self, testdataList, testlabelList):
        '''
        测试模型的准确率
        :param testdataList: 输入的测试数据集， list格式
        :param testlabelList:  输入测试集的label， list格式
        :return:  返回预测的准确率
        '''
        correct_num = 0

        for i in range(len(testlabelList)):
            # print(self.predict(testdataList[i]))
            if i % 100== 0:
                logging.info('Testing processing: ({}/{}) and the currect prediction:{}'.format(i, len(testdataList), correct_num))
            if self.predict(testdataList[i]) == testlabelList[i]:
                correct_num += 1
        return round(correct_num / len(testlabelList)* 100, 4)





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
    test_path = home + '/ML/mnist/mnist_test.csv'
    # train_path = home + '/ML/mnist/mnist_train_samples.csv'
    # test_path = home + '/ML/mnist/mnist_test_samples.csv'

    # 读取训练与测试集
    logging.info('Loading data....')

    traindataList, trainlabelList = loadData(train_path)
    testdataList, testlabelList = loadData(test_path)
    logging.info('Loading data done.')

    logging.info('Training the SVM model....')

    svm = SVM(traindataList[:1000], trainlabelList[:1000])


    svm.train()

    logging.info('Predicting one sample ....')
    prediction = svm.predict(testdataList[0])
    logging.info('The prediction and the ground truth is : ({}, {})'.format(prediction, testlabelList[0]))

    # 测试SVM算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    accuracy = svm.testModel(testdataList, testlabelList)

    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end - start), 4))





















