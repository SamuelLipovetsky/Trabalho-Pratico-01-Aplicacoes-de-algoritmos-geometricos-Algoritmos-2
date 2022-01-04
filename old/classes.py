

from random import randrange, seed, random
import heapq
import numpy
#this class can represent both a point or a hyperplane
#a leaf is always a point and the nodes in between are always hyperplanes
#if it is a hyperplane the "point" atribute represents the median
#of the dimension  that was split 
class node(object):
    def __init__(self,left , right ,point ,dim,is_leaf,id=None,cat=None):
        self.left = left 
        self.right = right 
        self.point = point 
        self.dim =dim
        self.is_leaf = is_leaf
        #id of the point
        self.id=id
        #category of the point 
        self.cat=cat
        
#simple euclidian distance function
def euc(a,b):
    sum=0
    for i in range(len(a)):
        temp = a[i] - b[i]
        sum += temp*temp
    return sum**0.5


class kdtree(object):   
    def __init__(self, points,dimension):    
        def __build(points,dimension,i):
            if len(points)==1:
                #create a leaf that represents a points
                return  node(None,None,points[0][:-1],i,True,points[0][-1][0],points[0][-1][1])
            else:
                #create a intermediate node that represents a hyperplane
                i= i%dimension
                points.sort(key=lambda x: x[i])
                half = len(points) >> 1
                v_left = __build(points[:half],dimension,i+1)
                v_right = __build(points[half:],dimension,i+1)
               
                return node(v_left,v_right,points[half][i],i,False)
    
        self.root = __build(points,dimension,0)   
    
class xnn(object):     
    def __init__(self,train):
        #constructing kdtree
        #assuming every point have the same number of dimensions
        
        self.root=kdtree(train,len(train[0])-1).root
        self.results={}
    def knear(self,node,test,k,heap):
        
            if node == None:
                return 
            if node.is_leaf :
                
                dist = euc( node.point,test )
                
                #using heapq as a min priority heap
                #by making the values negative
                #so the biggest value in the heap will be -heap[0]
                if len(heap)<k:
                    heapq.heappush(heap,( -dist,node.point,node.id,node.cat))
                    self.nearests = heap
                else:
                    if(dist < -heap[0][0]):
                        heapq.heappushpop(heap, (-dist,node.point,node.id,node.cat))
                        self.nearests = heap
                
                #chooses if the dimension of the test point
                #is greater or smaller than the median of the split hyper plane
                #this is a recursive search for leafs
            else:
             
                if(node.point > test[node.dim]):
                    
                    self.knear(node.left,test,k,heap)    
                else:
                    
                    self.knear(node.right,test,k,heap)     

                #instead of returning the best node found so far 
                #its checked if there a possibility of a  nearer neighbor
                #in a untested area
                radius = -heap[0][0]
                cp_to_plane =abs( test[node.dim] - node.point) 
                if ( radius <  cp_to_plane and len(heap)>k):   
                    return
                else:
                    if(node.point > test[node.dim]):
                        
                        self.knear(node.right,test,k,heap)
                    else:
                        
                        self.knear(node.left,test,k,heap)
    #runs knearest neighbor algorithm for every point in the test split
    #creates a dictionary with the id points as key and
    #the classification result and actual category of that point as values
    def knn_classifier(self,test,label_names,k):
        for  i in test:
      
            heap=[]
            self.nearests={}
            self.knear(self.root,i,k,heap)
    
            cat_counter=0
            label_0 = label_names[0]
            label_1 = label_names[1]
            #checks category of k nearest neighbors
            for j in self.nearests:
               
                if str(j[-1]) == str(label_0):
                    cat_counter+=1
            if cat_counter >  (k>>1) :
                self.results[i[-1][0]] ={"classification":label_0,"true_category":i[-1][1],"Nearests_neighbors":self.nearests}
            else:
                self.results[i[-1][0]] ={"classification":label_1,"true_category":i[-1][1],"Nearests_neighbors":self.nearests}
    
    #returns the precision and recall for the relevant label
    def get_stats(self,relevant_label):
        false_positives=0
        false_negatives=0
        true_positives =0
        true_negatives =0
        for j in self.results:
            i=self.results[j]
            print(i)
            if i["true_category"]==relevant_label:
                if i["classification"]== relevant_label:
                    true_positives+=1
                else:
                    false_negatives+=1
            else:
                if i["classification"]== relevant_label:
                    false_positives+=1
                else:
                    true_negatives+=1
    
     
        return (true_positives/(true_positives+false_positives),true_positives/(true_positives+false_negatives))
    
