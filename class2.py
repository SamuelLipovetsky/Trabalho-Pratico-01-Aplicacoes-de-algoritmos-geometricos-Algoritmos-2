

from random import randrange, seed, random
import heapq
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
                # print(points[0][:-1])
                return  node(None,None,points[0][:-1],i,True,points[0][-1][0],points[0][-1][1])
            else:
         
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
    #the regression result and actual category of that point 
    def knn_classifier(self,test,label_names,k):
        for  i in test:
      
            heap=[]
            
            self.knear(self.root,i,k,heap)
            cat_counter=0
            label_0 = label_names[0]
            label_1 = label_names[1]
            for j in self.nearests:
               
                if str(j[-1]) == str(label_0):
                    cat_counter+=1
            if cat_counter >  (k>>1) :
                self.results[i[-1][0]] ={"regression":label_0,"true_category":i[-1][1]}
            else:
                self.results[i[-1][0]] ={"regression":label_1,"true_category":i[-1][1]}
         

tree= xnn ([(2,3,(0,1)),(5,4,(1,0)),(9,6,(2,0)),(4,7,(3,0)),(8,1,(4,1)),(7,2,(5,0))])
tree.knn_classifier([(6,6,(6,1)),(1,2,(7,0)),(3,2,(8,1)),(4,5,(9,0))],[0,1],3)
print(tree.results)
