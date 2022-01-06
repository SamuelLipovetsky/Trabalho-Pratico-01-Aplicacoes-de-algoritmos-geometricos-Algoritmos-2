from xnn import xnn 
from aux_functions import keel_dat_reader
from aux_functions import split
import time


#dataset 1 -- pima
points = keel_dat_reader('./datasets/pima.dat',",")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["tested_positive","tested_negative"],20)
print("dataset 1 -- pima")
stats=tree.get_stats("tested_positive")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))

#dataset 2 -- monk-2
points = keel_dat_reader('./datasets/monk-2.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["1","0"],20)
print("dataset 2 -- monk-2")
stats=tree.get_stats("1")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))

#dataset 3 -- bupa
points = keel_dat_reader('./datasets/bupa.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["2","1"],30)
print("dataset 3 -- bupa")
stats=tree.get_stats("1")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))

#dataset 4 -- haberma

points = keel_dat_reader('./datasets/haberman.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["positive", "negative"],10)
print("dataset 4 -- haberman")
stats=tree.get_stats("positive")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))

# dataset 5 -- sonar (largest number of dimensions)

points = keel_dat_reader('./datasets/sonar.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["M", "R"],20)
print("dataset 5 -- sonar")
stats=tree.get_stats("R")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))

# dataset 6 -- banana

start = time.time()
points = keel_dat_reader('./datasets/banana.dat',",")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["-1.0","1.0"],20)
print("dataset 6 -- banana")
stats=tree.get_stats("1.0")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))

# dataset 7 -- phoneme

points = keel_dat_reader('./datasets/phoneme.dat',",")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["0","1"],10)
print("dataset 7 -- phoneme")
stats=tree.get_stats("1")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))
    
# dataset 8 -- appendicitis

points = keel_dat_reader('./datasets/appendicitis.dat',",")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["0","1"],10)
print("dataset 8 --  appendicitis")
stats=tree.get_stats("1")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))
    
# dataset 9 -- titanic

points = keel_dat_reader('./datasets/titanic.dat',",")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["-1.0","1.0"],20)
print("dataset 9 --  titanic")
stats=tree.get_stats("1.0")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))
    
# dataset 10 -- wdbc
 
points = keel_dat_reader('./datasets/wdbc.dat',",")
pct =int(len(points)*0.9)
train,test = points[:pct],points[pct:]
tree=xnn(train)
tree.knn_classifier(test,["M","B"],10)
print("dataset 10 --  wdbc")
stats=tree.get_stats("M")
print("precision:" + str(stats[0]), 
    "recall:" + str(stats[1]),
    "accuracy:" +str(stats[2]))

