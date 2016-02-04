# Perceptron Learning Algorithm
# see 02_handout.pdf at http://www.csie.ntu.edu.tw/~htlin/mooc/

from __future__ import division
import ConfigParser
import matplotlib.pyplot as plt

# number of training data
n = -1
threshhold = 0
inputs = None
outputs = None
dimension = None

def getX2FromX1(x1, w):
    return (-w[0] - w[1] * x1) / w[2]

def plotGraph(inputs, outputs, w):
    lBound = -7
    rBound = 7
    xs = getXs(inputs)
    ys = getYs(inputs)
    for i in range(len(outputs)):
        if outputs[i] == 1:
            plt.plot(xs[i], ys[i], 'ro')
        elif outputs[i] == -1:
            plt.plot(xs[i], ys[i], 'bx')
    # plot line
    plt.plot([lBound, rBound], [getX2FromX1(lBound, w), getX2FromX1(rBound, w)], 'k')
    plt.axis([lBound, rBound, lBound, rBound])
    plt.grid(True)
    plt.xlabel('Annual Income')
    plt.ylabel('Total Assets')
    plt.show()

def getXs(points):
    length = len(points)
    xs = [None] * length
    for i in range(length):
        xs[i] = points[i][1]
    return xs

def getYs(points):
    length = len(points)
    ys = [None] * length
    for i in range(length):
        ys[i] = points[i][2]
    return ys

def mult(vec1, vec2):
    # TODO: make sure vec1 and vec2 are of the same size
    result = 0
    for i in range(len(vec1)):
        result += (vec1[i] * vec2[i])
    return result

def init():
    threshhold = 1

def parseData():
    # parse training data
    Config = ConfigParser.ConfigParser()
    #Config.read('G:\\MazeMyth\\Research\\MachineLearning\\data.ini')
    Config.read('data.ini')
    sections = Config.sections()
    tmpInputs = Config.options(sections[0])
    global dimension
    # x0 is not counted for
    dimension = len(tmpInputs[0].split(','))
    global n
    n = len(tmpInputs)
    global inputs
    global outputs
    inputs = [[0 for x in range(dimension+1)] for x in range(n)]
    outputs = [0 for x in range(n)]
    for i in range(n):
        tmpStr = tmpInputs[i].split(',')
        tmpInt = map(int, tmpStr)
        outputs[i] = int(Config.get(sections[0], tmpInputs[i]))
        for j in range(dimension+1):
            if j == 0:
                inputs[i][j] = 1     # this stands for x0
            else:
                inputs[i][j] = tmpInt[j-1]

if __name__ == "__main__":
    init()
    parseData()
    print('dimension: ' + `dimension`)
    # initial hypothesis
    w = [None] * (dimension + 1)
    for i in range(dimension + 1):
        if i == 0:
            w[i] = -threshhold
        else:
            w[i] = 0

    print('Initial w: ' + `w`)
    # applying PLA
    while True:
        # find a mistake
        allCorrect = True
        for i in range(n):
            result = 1 if mult(inputs[i], w) > 0 else -1
            if(result != outputs[i]):
                print(`inputs[i]` + ' x ' + `w` + ' != ' + `outputs[i]`)
                # correct this vector
                # we can't update w0, since w0 stands for thresh hold
                for j in range(1, dimension + 1):
                    w[j] = w[j] + outputs[i] * inputs[i][j]
                allCorrect = False
                print('Update w as ' + `w`)
        if allCorrect:
            break

    plotGraph(inputs, outputs, w)
