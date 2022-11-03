import numpy as np
import json
#
# Evaluate  the  accuracy  of  a  decision  tree.  It  contains  a  function,  main(fname), 
# where fname is the file in which a (full or pruned) decision tree is saved.  
# The function returns the accuracy of the decision tree when running on the test set.
#
# Create  evaluate.py,  and  run  evaluate.main(fname1)  where  fname1  is  the  file  name  in 
# which you save your fully-grown decision tree, and print the accuracy in the IDLE shell
#

def predict(tr, r, li): # using same functions from pruneTree class
    if type(tr) != list:
        return tr
    a = tr.copy()
    
    while type(a) == list:
        for i in range(len(li)):
            if a[0] == li[i][0]:
                break   
        a = a[1][str(r[i-1])]
    return a

def accuracy(tr, m, li): # using same functions from pruneTree class
    ac = 0
    for i in range(np.size(m,axis=1)):
        if predict(tr, m[1:,i], li) == m[0,i]:
                   ac += 1
    return ac/np.size(m,axis=1)


def main(fname):
    with open('../data/'+fname) as f: # bring in the tree -- fname is filename of tree being used
        tree = json.load(f)

    with open('../data/dataDesc.txt') as f:
        li = json.load(f)

    testMatrix = np.loadtxt("../data/test.txt", dtype=int) # bring in the test data matrix
    
    acc = accuracy(tree, testMatrix, li)
    
    print("The accuracy of decision tree at <"+fname+"> is", acc*100,"%")

if __name__ == "__main__":
    main('tree.txt')