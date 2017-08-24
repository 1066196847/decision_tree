# coding=utf-8
import csv
import os
import pickle
import cPickle
from math import ceil
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame

'''
函数说明：绘制一个树节点
'''
import matplotlib.pyplot as plt
# 定义文本框、箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


'''
函数说明：要绘制分裂图，必须要知道树的深度（树的层数）、叶节点的数目
输入：tree（在decision_tree.py这个函数返回的“树的结构”，一个字典类型变量）
返回值：树的叶节点数目、树深度
tree = {'wenli':
{1: {'gendi':
           {1: 1,
            2: {'seze':
                     {1: 1,
                      2: {'chugan': {1: 1, 2: 0}}}},
            3: 0}},
 2: {'chugan': {1: 0, 2: 1}},
 3: 0}}
'''
def getNumLeafs(tree):
    num = 0 #用来存储叶子节点的个数
    firstStr = tree.keys()[0] #某个特征的第一个“属性值”
    secondDict = tree[firstStr] #字典中，第一个“属性值”是一个索引，求出它对应的值
    for key in secondDict.keys(): #依次处理每一个索引对应的值
        if type(secondDict[key]).__name__ == 'dict':
            num += getNumLeafs(secondDict[key])
        else:
            num += 1
    return num
def getTreeDepth(tree):
    depth = 0
    firstStr = tree.keys()[0]  # 某个特征的第一个“属性值”
    secondDict = tree[firstStr]  # 字典中，第一个“属性值”是一个索引，求出它对应的值
    for key in secondDict.keys():  # 依次处理每一个索引对应的值
        if type(secondDict[key]).__name__ == 'dict':
            this_depth = 1 + getTreeDepth(secondDict[key])
        else:
            this_depth = 1
        if(this_depth > depth):
            depth = this_depth
    return depth


'''
函数说明：在父子两个节点坐标，的中间位置，填充几个文字（不带边框）
输入：cntrPt（子节点的坐标），parentPt（父节点的坐标），txtString（文本信息）
'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

'''
函数说明：绘制一条线段 + 一个箭头 + 一个边框（里面带有文字nodeTxt），
输入：nodeTxt（特征名），centerPt（中心节点），parentPt（父节点），nodeType（decisionNode leafNode 决定边框、文字大小）
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center",
                            ha="center", bbox=nodeType, arrowprops=arrow_args)

'''
函数说明：画出来一幅图，就是决策树分裂的图
输入：myTree（字典结构的分裂图），parentPt（父节点的坐标），nodeTxt（文本内容）
重难点：这个函数的逻辑很简单，碰到“非叶子节点”，就用递归招呼；碰到叶子节点，就用 画出叶子节点的逻辑招呼。难在 每一个框图的坐标布置上！
'''
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    # 第一个分裂“特征”
    firstStr = myTree.keys()[0]
    # 得到接下来要分裂 节点的坐标
    cntrPt = (plotTree.xOff + ( 1.0 + float(numLeafs) )/2.0/plotTree.totalW, plotTree.yOff )
    # 在 cntrPt parentPt 中间位置处，打印文字信息--nodeTxt
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 画出来 一个线段 + 一个箭头 + 一个边框（里面带文字firstStr）
    plotNode(firstStr, cntrPt, parentPt, decisionNode)

    # 第一个分裂“特征”对应的字典是什么？myTree[firstStr]
    secondDict = myTree[firstStr]
    # 更新树中，下一个节点的“y坐标”
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':# 如果不是叶子节点，就进入递归
            plotTree(secondDict[key], cntrPt, str(key) )
        else: # 如果是叶子节点，就进画出来叶子节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD





def createPlot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # 上面两行代码是“建立”一个空白框，等待下面的代码再往上面“添加”一些东西
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) # frameon=False 不要边界线，只显示点所在的“短线”
    # 给 plotTree添加两个成员变量。totalW（存储树的宽度），toatlD（存储树的深度）
    # 利用这两个变量，可以将“树”合理的放在最中心的位置！
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    # 给 plotTree添加两个成员变量。xOff yOff用于追踪已经绘制的节点位置，以及防止下一个节点的适当位置
    plotTree.xOff = -0.5/plotTree.totalW #假设有8个叶子节点，plotTree.xOff就会被设置为 -0.0625
    plotTree.yOff = 1.0

    plotTree(tree, (0.5,1.0), '')
    plt.show()










if __name__ == "__main__":
    tree = {'wenli': {1: {'gendi': {1: 1, 2: {'midu': {'small_0.3815': 0, 'big_0.3815': 1}}, 3: 0}}, 2: {'chugan': {1: 0, 2: 1}}, 3: 0}}
    # a = getNumLeafs(tree)
    # print(a)
    # b= getTreeDepth(tree)
    # print(b)

    createPlot(tree)












































