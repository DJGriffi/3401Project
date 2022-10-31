import json
import math
import numpy as np

def main():
    # matrix = np.loadtxt("../data/train.txt")
    matrix = np.loadtxt("/media/data/DrewSchool/3401Project/decision-tree/data/train.txt")

    # tree = 
    # print(matrix.shape)

    # print(matrix[1][0])

    ageMatrix = matrix[[0]]
    ageMatrix2 = matrix[[1]]
    ageMatrix3 = matrix[[0,1],:]
    ageMatrix4 = matrix[:,[0,1]]
    ageMatrix5 = np.stack((matrix[[0]], matrix[[2]]),axis=2)
    # print(ageMatrix4)
    # print(ageMatrix5)
    # print(len(matrix))
    # print(ageMatrix)
    # print(ageMatrix2)
    # print(ageMatrix2.size)

    # unique_values, occurrence_count = np.unique(ageMatrix5, return_counts=True, axis= 1)
    # print(unique_values)
    # print(occurrence_count)

    # with open("../data/dataDesc.txt") as f:
    with open("/media/data/DrewSchool/3401Project/decision-tree/data/dataDesc.txt") as f: 
        attributeList = json.load(f)
    
    # print(attributeList)

    # m1 = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]])
    # a1 = [['RISK', [1,2]]]
    tree = generateTree(matrix, attributeList)
    # probabilities = occurrence_count / len(ageMatrix5[0])
    # print(probabilities)
    # print(calculateEntropy(ageMatrix))
    # print(calculateEntropy(ageMatrix5))

    return 0

def calculateEntropy(M, uniqueAttrValues, uniqueAttrOccur):
    unique_values, occurrence_count = np.unique(M, return_counts = True, axis= 1)
    print(unique_values)
    print(occurrence_count)
    print()
    print(uniqueAttrValues)
    print(uniqueAttrOccur)
    print()
    entropies = []
    for i in range(0, len(uniqueAttrValues)):
        probabilities = []
        probabilities.append(occurrence_count[i]/uniqueAttrOccur[i])
        probabilities.append(occurrence_count[i + len(uniqueAttrValues)]/uniqueAttrOccur[i])
        entropies.append(-np.sum([p * math.log(p,2) for p in probabilities if p > 0]))

    weightedEntropy = 0
    for i in range(0, len(uniqueAttrValues)):
        weightedEntropy += ((uniqueAttrOccur[i]/np.sum(occurrence_count)) * entropies[i])

    # probabilities = occurrence_count / len(M[0])
    
    return weightedEntropy

def generateTree(matrix, attributeList):
    node = []
       
    unique_values, occurrence_count = np.unique(matrix[[0]], return_counts=True)
 
    if (len(unique_values) == 1):
        print("here1")
        return unique_values
    elif (len(attributeList) == 1):
        if (occurrence_count[0] > occurrence_count[1]):
            print("here2")
            return [unique_values[0]]
        else:
            print("here3")
            return [unique_values[1]]

    entropies = []  

    for i in range(1, len(matrix)):
        m = np.stack((matrix[[0]], matrix[[i]]),axis=2)
        unique_values, occurrence_count = np.unique(matrix[[i]], return_counts=True)
        entropies.append(calculateEntropy(m, unique_values, occurrence_count))

    minEntropyIndex = np.argmin(entropies) + 1
    testAttributeRow = matrix[[minEntropyIndex],:].tolist()
    matrix = np.delete(matrix, minEntropyIndex, 0)
    print(matrix)
    node.append(attributeList[minEntropyIndex][0])
    # node.append({})
    # testAttributeDomain = [[],[1]]
    testAttributeDomain = attributeList.pop(minEntropyIndex)
    
    print(len(testAttributeRow[0]))
    # print(attributeList)
    # print(entropies)
    # print(attributeList)
    # print(np.argmin(entropies))
    print(node)

    dict = {}
    for child in testAttributeDomain[1]:
        print("COPIED MATRIX")
        matrixCopy = matrix.copy()
        removeIndex = []
        print(matrixCopy.shape)
        for i in range(0, len(testAttributeRow[0])):
            if (testAttributeRow[0][i] == child):
                # matrixCopy.append(matrix[:,[i]].tolist())
                removeIndex.append(i)
        matrixCopy = np.delete(matrixCopy, removeIndex, 1)
        print(matrixCopy.shape)
        # matrixArr = np.array(matrixCopy)
        # matrixArr = matrixArr.T
        # matrixArr = np.delete(matrixArr, 0 , 0)
        # matrixShape = matrixArr.shape
        # print(matrixShape[1])
        # np.reshape(matrixArr, (matrixShape[1], matrixShape[2]))
        # print(matrixArr.shape)
        
        childNode = generateTree(matrixCopy, attributeList)
        dict.update({child: childNode})

    node.append(dict)
    return node
        # print(matrixArr)

if __name__ == "__main__":
    main()