import json
import math
import numpy as np

def main():
    matrix = np.loadtxt("../data/train.txt")
    # matrix = np.loadtxt("/media/data/DrewSchool/3401Project/decision-tree/data/train.txt")
    global dataLen
    dataLen = len(matrix[0])
    # with open("/media/data/DrewSchool/3401Project/decision-tree/data/dataDesc.txt") as f: 
    with open("../data/dataDesc.txt") as f:
        attributeList = json.load(f)

    tree = generateTree(matrix, attributeList)
    
    with open("../data/tree.txt", "w") as f:
        json.dump(tree, f)

    return 0

    
def calculateEntropy(M, uniqueAttrOccur, domain):
    """ Calculates the weighted entropy of a candidate test attribute from a given data set and class variable.

    Args:
        M (ndarray):a matrix containing class label and attribute value pairs. For example, if the class
        label domain is {1,2} and the attribute domain is {a,b,c}, M could look like this:
        [[[1 a]
          [1 b]
          [2 c]
          [1 c]]]
        uniqueAttrOccur (ndarray): an array containing the number of occurrences of each attribute value in M,
        i.e. [23 54 7]
        domain (list): list of the current candidate test attribute's domain
  

    Returns:
        float: the weighted entropy of the current candidate test attribute. 
    """

    unique_values, occurrence_count = np.unique(M, return_counts = True, axis= 1) #get the unique pairings and occurrence count
    
    entropies = []
   
    for i in range(0, len(uniqueAttrOccur)):
        probabilities = []

        if (len(occurrence_count) != (len(uniqueAttrOccur) * 2)): # check if each attribute domain value is paired with each 
                                                                  # possible class label value and if not, remove them
            # domain = attributeList[n][1]
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
        # weightedEntropy += ((uniqueAttrOccur[i]/np.sum(occurrence_count)) * entropies[i])
        weightedEntropy += ((uniqueAttrOccur[i]/dataLen) * entropies[i])
        # print(np.sum(occurrence_count))
    print(weightedEntropy)
    return weightedEntropy


def generateTree(matrix, attributeList):
    """Generate a sequence to represent a decision tree from a given data set and list
    of attributes with their associated domains

    Args:
        matrix (ndarray): a training data set 
        attributeList (list): list of attributes with their associated domains.

    Returns:
        list: A sequence representing a decision tree
    """
    node = []  
    unique_values, occurrence_count = np.unique(matrix[[0]], return_counts=True) #get unique values and their associated occurrences for the class label
   

    if (len(unique_values) == 1): #test the stopping conditions
        return int(unique_values[0])

    elif (len(attributeList) == 1):
        if (occurrence_count[0] > occurrence_count[1]):
            return int(unique_values[0])
          
        else:
            return int(unique_values[1])

    entropies = []  

    for i in range(1, len(matrix)):
        m = np.stack((matrix[[0]], matrix[[i]]),axis=2) #pair the class label with a candidate test attribute
        unique_values, occurrence_count = np.unique(matrix[[i]], return_counts=True) #get unique values and occurrences of the candidate test attribute's 
        domain = attributeList[i][1]                                                 #domain values.
        entropies.append(calculateEntropy(m, occurrence_count, domain))

    minEntropyIndex = np.argmin(entropies) + 1 #select the test attribute with the lowest entropy, therefore providing the greatest information gain
    testAttributeRow = matrix[[minEntropyIndex],:].tolist()
    matrix = np.delete(matrix, minEntropyIndex, 0) #remove the selected test attribute from the training data set
    node.append(attributeList[minEntropyIndex][0])
    testAttributeDomain = attributeList.pop(minEntropyIndex) #remove the selected test attribute from the attribute list

    dict = {}
    for child in testAttributeDomain[1]: #for each value in the selected test attribute grow the decision tree.
        matrixCopy = matrix.copy()
        removeIndex = []

        for i in range(0, len(testAttributeRow[0])): 
            if (testAttributeRow[0][i] != child): #find the columns that dont have the child's value and remove them from the
                removeIndex.append(i)             #characteristic data set
        matrixCopy = np.delete(matrixCopy, removeIndex, 1)       
        childNode = generateTree(matrixCopy, attributeList)
        dict.update({child: childNode})

    node.append(dict)
    return node

if __name__ == "__main__":
    main()