# -*- coding:utf-8 -*-
# @time :2020/7/25
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

'''
实现朴素贝叶斯分类器
并采用mnist数据集测试模型
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

        labelArr.append(int(curLine[0]))
        # 进行二值化处理，将大于128的标记为1， 小于128的标记为0
        dataArr.append([int(int(num)>128) for num in curLine[1:]])
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)

    # 返回data和label
    return dataArr, labelArr


class NavieBayes:
    def __init__(self, num_classes, num_features, traindataArr, trianlabelArr):
        '''
        初始化朴素贝叶斯分类器类
        :param num_classes:  类别数目
        :param num_features:  特征维度
        :param traindataArr:  训练集
        :param trianlabelArr: 训练集标签
        '''
        self.num_classes = num_classes
        self.num_features = num_features
        self.traindataArr, self.trainlabelArr = traindataArr, trainlabelArr
        self.py, self.px_y = self.getProbability()



    def naviebayes(self, x):
        '''
        利用朴素贝叶斯进行概率估计
        :param py: 先验概率
        :param pxy: 条件概率
        :param x: 待测样本点
        :return: 返回类别
        '''
        p= [0] * self.num_classes

        # 计算每个类别的概率
        for i in range(self.num_classes):
            # 由于在getProbaility中计算得到的概率值已经经过了log运算，因此这里的概率值可以采用连加的形式
            sum = 0
            for j in range(self.num_features):
                sum += self.px_y[i][j][x[j]]
            p[i] = sum + self.py[i]
        return p.index(max(p))


    def getProbability(self):
        '''
        计算所有训练集的先验与条件概率
        :param dataArr:  输入的训练样本集（list格式）
        :param labelArr:  输入的训练样本的label （list格式）
        :return:  返回训练集的先验概率分布与条件概率分布
        '''

        # 首先计算先验分布py，初始化py数组
        py = np.zeros((self.num_classes, 1))

        for i in range(self.num_classes):
            # 不考虑出现概率值为0的情况
            # np.mat(self.trainlabelArr == i)会让对应与等于i的为True， 不等的为False
            # py[i] = np.sum(np.mat(self.trainlabelArr == i)) / (len(self.trainlabelArr))

            # 考虑概率值为0的情况，采用laplace平滑
            py[i] = np.sum(np.mat(self.trainlabelArr == i) + 1) / (len(self.trainlabelArr) + self.num_classes)

        # 最后求后验概率估计的时候，形式是各项的相乘（“4.1 朴素贝叶斯法的学习” 式4.7），这里存在两个问题：1.某一项为0时，结果为0.
        # 这个问题通过分子和分母加上一个相应的数可以排除，前面已经做好了处理。2.如果特诊特别多（例如在这里，需要连乘的项目有784个特征
        # 加一个先验概率分布一共795项相乘，所有数都是0-1之间，结果一定是一个很小的接近0的数。）理论上可以通过结果的大小值判断， 但在
        # 程序运行中很可能会向下溢出无法比较，因为值太小了。所以人为把值进行log处理。log在定义域内是一个递增函数，也就是说log（x）中，
        # x越大，log也就越大，单调性和原数据保持一致。所以加上log对结果没有影响。此外连乘项通过log以后，可以变成各项累加，简化了计算。
        py = np.log(py)

        logging.info('Getting the prior distribution.')


        # 计算条件概率分布pxy，初始化pxy数组
        # 一共有num_classes类，一共有num_features个特征， 每个特征有两种取值，1或者0
        px_y = np.zeros((self.num_classes, self.num_features, 2))

        # 对标记集进行遍历
        for i in range(len(self.trainlabelArr)):
            # 获取当前循环所使用的标记
            label = self.trainlabelArr[i]
            # 获取当前要处理的样本
            x = self.traindataArr[i]
            # 对该样本的每一维特诊进行遍历
            for j in range(self.num_features):
                # 在矩阵中对应位置加1
                # 这里还没有计算条件概率，先把所有数累加，全加完以后，在后续步骤中再求对应的条件概率
                px_y[label][j][x[j]] += 1

        for label in range(self.num_classes):
            for j in range(self.num_features):
                # 分别计算第j个特征为0和1的个数
                px_y0 = px_y[label][j][0]
                px_y1 = px_y[label][j][1]

                # 计算条件概率
                px_y[label][j][0] = np.log((px_y0 +1) / (px_y0 + px_y1 + 2))
                px_y[label][j][1] = np.log((px_y1 +1) / (px_y0 + px_y1 + 2))
        logging.info('Getting the Conditional probability distribution.')

        return py, px_y

    def testModel(self,dataArr, labelArr):
        '''
        利用测试集测试训练集的
        :param py: 先验概率分布
        :param pxy: 条件概率分布
        :param dataArr: 测试集数据
        :param labelArr: 测试集的label
        :return: 返回准确率
        '''
        correct_num = 0
        for i in range(len(dataArr)):
            if i %50 == 0:
                logging.info('Testing the testdata: ({}/{}).'.format(i, len(labelArr)))

            label = self.naviebayes(dataArr[i])
            if label == labelArr[i]:
                correct_num += 1
        return round(correct_num / len(labelArr), 4)






if __name__ == '__main__':

    # 定义一个日志模块来保存日志
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='NaiveBayes.log',
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

    traindataArr, trainlabelArr =loadData(train_path)
    testdataArr, testlabelArr = loadData(test_path)
    logging.info('Loading data done.')

    num_classes = 10
    num_features = 28 * 28
    Naviebayes = NavieBayes(num_classes, num_features,traindataArr, trainlabelArr)



    #测试朴素贝页斯算法的准确率
    # 挑选测试集的前200个进行测试，防止运行时间过长
    accuracy = Naviebayes.testModel(testdataArr[:200], testlabelArr[:200])


    end = time.time()

    logging.info('accuracy:{}'.format(accuracy))
    logging.info('Total Time: {}'.format(round(end-start), 4))
