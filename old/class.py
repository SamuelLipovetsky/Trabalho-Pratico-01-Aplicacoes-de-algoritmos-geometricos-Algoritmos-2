

from random import randrange, seed, random
import heapq
#this class can represent both a point or a hyperplane
#a leaf is always a point and the nodes in between are always hyperplanes
#if it is a hyperplane the "point" atribute represents the median
#of the dimension  that was split 
class node(object):
    def __init__(self,left , right ,point ,dim,is_leaf):
        self.left = left 
        self.right = right 
        self.point = point 
        self.dim =dim
        self.is_leaf = is_leaf
        
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
                # print(points[0])
                return  node(None,None,points[0],i,True)
            else:
                # i= (i+1)%dimension
                i= i%dimension
                points.sort(key=lambda x: x[i])
                
                half = len(points) >> 1
            
               
                v_left = __build(points[:half],dimension,i+1)
                v_right = __build(points[half:],dimension,i+1)
                return node(v_left,v_right,points[half][i],i,False)
    
        self.root = __build(points,dimension,0)   

  
       
    def knearest(self,test,k) :
        def __knear(node,test,k,heap):
            if node == None:
                return 
            if node.is_leaf :
                dist = euc( node.point,test )
                
                #using heapq as a min priority heap
                #by making the values negative
                #so the biggest value in the heap will be -heap[0]
                if len(heap)<k:
                    heapq.heappush(heap,( -dist,node.point))
                else:
              
                    if(dist < -heap[0][0]):
                        heapq.heappushpop(heap, (-dist,node.point))
                
                #chooses if the dimension of the test point
                #is greater or smaller than the median of the split hyper plane
                #this is a recursive search for leafs
            else:
                if(node.point > test[node.dim]):
                    
                    __knear(node.left,test,k,heap)  
                else:
                    
                    __knear(node.right,test,k,heap)     

                #instead of returning the best node found so far 
                #its checked if there a possibility of a  nearer neighbor
                #in a untested area
                radius = -heap[0][0]
                cp_to_plane =abs( test[node.dim] - node.point) 
                if ( radius <  cp_to_plane ):   
                    return
                else:
                    if(node.point > test[node.dim]):
                        
                        __knear(node.right,test,k,heap)
                    else:
                        
                        __knear(node.left,test,k,heap)
               

        heap=[]
        heapq.heapify(heap)
        __knear(self.root,test,k,heap)      
        for i in range(len(heap)):
            print(heap[i])
                



    def print(self):
        def print_h(Node):
            if (Node.is_leaf):
                print("leaf" , Node.point)
            else:
                print("mediana :" , Node.point ,"dimensao :",Node.dim)
               
                print_h(Node.left)
                print_h(Node.right)
          

        print_h(self.root)

# tree= kdtree ([(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)],2)
# tree = kdtree ([(1,1),(2,2),(3,3),(4,4),(5,5)],2)

# a=[(100,100),(101,102),(103,104),(105,105),(110,110)]
a=[(2,3,1,2,3,0,1,0),(5,4,1,2,3,0,1,0),(9,6,1,2,3,0,1,0),(4,7,1,2,3,0,1,0),(8,1,1,2,3,0,1,0),(7,2,1,2,3,0,1,0)]
for i in range(5000):
    a.append((randrange(90,100,1),randrange(90,100,2),randrange(90,100,1),randrange(90,100,2),randrange(90,100,1),randrange(90,100,2) ,randrange(90,100,1),randrange(90,100,2)))



tree =kdtree(a,8) 
# tree.print()
tree.knearest((10,20,1,2,3,0,1,0),6)