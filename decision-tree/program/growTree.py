import json
import math
import numpy as np

def main():
    matrix = np.loadtxt("../data/train.txt")
    # matrix = np.loadtxt("/media/data/DrewSchool/3401Project/decision-tree/data/train.txt")

    # with open("/media/data/DrewSchool/3401Project/decision-tree/data/dataDesc.txt") as f: 
    with open("../data/dataDesc.txt") as f:
        attributeList = json.load(f)

    tree = generateTree(matrix, attributeList)
    
    with open("../data/tree.txt", "w") as f:
        json.dump(tree, f)

    return 0


def calculateEntropy(M, uniqueAttrValues, uniqueAttrOccur, attributeList, n):
    unique_values, occurrence_count = np.unique(M, return_counts = True, axis= 1)
    
    entropies = []
   
    for i in range(0, len(uniqueAttrOccur)):
        probabilities = []

        if (len(occurrence_count) != (len(uniqueAttrOccur) * 2)):
            domain = attributeList[n][1]
            index = []
            uniqueList = unique_values.tolist()
            indexToDelete = 1

            for v in domain:
                count = 0

                for j in range(0, len(unique_values[0])):
                    if (uniqueList[0][j][1] == v):
                        count += 1
                        valIndex = j

                if (count != 2):
                    index.append(valIndex)
                    uniqueAttrOccur = np.delete(uniqueAttrOccur, v-indexToDelete)
                    indexToDelete += 1

            unique_values = np.delete(unique_values, index, 1)
            occurrence_count = np.delete(occurrence_count, index, 0)

            for k in range(0, len(uniqueAttrOccur)):
                probabilities.append(occurrence_count[k]/uniqueAttrOccur[k])
                probabilities.append(occurrence_count[k + len(uniqueAttrOccur)]/uniqueAttrOccur[k])
                entropies.append(-np.sum([p * math.log(p,2) for p in probabilities if p > 0]))

            break

        else:     
            probabilities.append(occurrence_count[i]/uniqueAttrOccur[i])
            probabilities.append(occurrence_count[i + len(uniqueAttrOccur)]/uniqueAttrOccur[i])
            entropies.append(-np.sum([p * math.log(p,2) for p in probabilities if p > 0]))

    weightedEntropy = 0

    for i in range(0, len(uniqueAttrOccur)):
        weightedEntropy += ((uniqueAttrOccur[i]/np.sum(occurrence_count)) * entropies[i])
    
    return weightedEntropy

def generateTree(matrix, attributeList):
    node = []  
    unique_values, occurrence_count = np.unique(matrix[[0]], return_counts=True)
   
    if (len(unique_values) == 1):
        return int(unique_values[0])

    elif (len(attributeList) == 1):
        if (occurrence_count[0] > occurrence_count[1]):
            return int(unique_values[1])
          
        else:
            return int(unique_values[0])

    entropies = []  

    for i in range(1, len(matrix)):
        m = np.stack((matrix[[0]], matrix[[i]]),axis=2)
        unique_values, occurrence_count = np.unique(matrix[[i]], return_counts=True)
        entropies.append(calculateEntropy(m, unique_values, occurrence_count, attributeList, i))

    minEntropyIndex = np.argmin(entropies) + 1
    testAttributeRow = matrix[[minEntropyIndex],:].tolist()
    matrix = np.delete(matrix, minEntropyIndex, 0)
    node.append(attributeList[minEntropyIndex][0])
    testAttributeDomain = attributeList.pop(minEntropyIndex)

    dict = {}
    for child in testAttributeDomain[1]:
        matrixCopy = matrix.copy()
        removeIndex = []

        for i in range(0, len(testAttributeRow[0])):
            if (testAttributeRow[0][i] == child):
                removeIndex.append(i)
        matrixCopy = np.delete(matrixCopy, removeIndex, 1)       
        childNode = generateTree(matrixCopy, attributeList)
        dict.update({child: childNode})

    node.append(dict)
    return node

if __name__ == "__main__":
    main()