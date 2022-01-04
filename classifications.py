from xnn import xnn 
from aux_functions import keel_dat_reader
from aux_functions import split


#dataset 1 -- pima
points = keel_dat_reader('./datasets/pima.dat',",")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["tested_positive","tested_negative"],20)
print("dataset 1 -- pima")
stats=tree.get_stats("tested_positive")
print("precision:" + str(stats[0]), "recall:" + str(stats[1]))

#dataset 2 -- monk-2

points = keel_dat_reader('./datasets/monk-2.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["1","0"],20)
print("dataset 2 -- monk-2")
stats=tree.get_stats("1")
print("precision:" + str(stats[0]), "recall:" + str(stats[1]))

#dataset 3 -- bupa
points = keel_dat_reader('./datasets/bupa.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["2","1"],20)
print("dataset 3 -- bupa")
stats=tree.get_stats("1")
print("precision:" + str(stats[0]), "recall:" + str(stats[1]))

#dataset 4 -- haberma

points = keel_dat_reader('./datasets/haberman.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["positive", "negative"],20)
print("dataset 4 -- haberman")
stats=tree.get_stats("positive")
print("precision:" + str(stats[0]), "recall:" + str(stats[1]))

#dataset 5 -- sonar (largest number of dimensions)

points = keel_dat_reader('./datasets/sonar.dat',", ")
train,test = split(points,0.7)
tree=xnn(train)
tree.knn_classifier(test,["M", "R"],20)
print("dataset 5 -- sonar")
stats=tree.get_stats("R")
print("precision:" + str(stats[0]), "recall:" + str(stats[1]))



