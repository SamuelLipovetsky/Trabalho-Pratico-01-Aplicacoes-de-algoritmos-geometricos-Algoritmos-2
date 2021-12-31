

from random import seed, random
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
        def __build(points,dimension,i=0):
            if len(points)==1:
                # print(points[0])
                return  node(None,None,points[0],i,True)
            else:
                # i= (i+1)%dimension
                i= i%dimension
                points.sort(key=lambda x: x[i])
                
                half = len(points) >> 1
                # print(points[:half],points[half],points[half:],i)
               
                v_right = __build(points[:half],dimension,i+1)
                v_left = __build(points[half:],dimension,i+1)
                return node(v_left,v_right,points[half][i],i,False)
    
        self.root = __build(points,dimension)   

    def nearest(self, test):
        def __near(node,test,closest_p,closest_d):
            
            if node == None:
                return (closest_p,closest_d)   
            if node.is_leaf :
                
                dist = euc( node.point,test )
                # print(node.point)
                if(dist < closest_d):
                    closest_p = node.point
                    closest_d =dist
                    return (closest_p,closest_d)                   
                else:
                    return(closest_p,closest_d)   
            else:
                #chooses if the dimension of the test point
                #is greater or smaller than the median of the split hyper plane

                if(node.point >= test[node.dim]):
                    
                    p=__near(node.left,test,closest_p,closest_d)
                else:
                    
                    p= __near(node.right,test,closest_p,closest_d)
                closest_p =p[0]
                closest_d =p[1]
                radius =  abs( closest_p[node.dim]- test[node.dim])
                cp_to_plane =abs( closest_p[node.dim]- node.point) 
                if (radius < cp_to_plane ):
                    return p
                if(node.point >= test[node.dim]):
                    
                    p=__near(node.right,test,closest_p,closest_d)
                else:
                    
                    p= __near(node.left,test,closest_p,closest_d)
                
                return p
        #fix this
        temp_point=(0,0)
        print(__near(self.root,test,temp_point,100000000))
       
    def knearest(self,test,k) :
        def __knear(node,test,k,heap):
            if node == None:
                return 
            if node.is_leaf :
                dist = euc( node.point,test )
                # print(node.point)
                if len(heap)<k:
                    heapq.heappush(heap,( node.point,-dist))
                if(dist < -heap[-1][1]):
                    heapq.heappushpop(heap, (node.point,-dist))
            else:
                #chooses if the dimension of the test point
                #is greater or smaller than the median of the split hyper plane

                if(node.point >= test[node.dim]):
                    
                    __knear(node.left,test,k,heap)
                else:
                    
                    __knear(node.right,test,k,heap)  
                closest_p = heap[-1][0]
                closest_d = heap[-1][1]
                radius =  abs( closest_p[node.dim]- test[node.dim])
                cp_to_plane =abs( closest_p[node.dim]- node.point) 
                if (radius < cp_to_plane ):
                    return 
                if(node.point >= test[node.dim]):
                    
                    __knear(node.right,test,k,heap)
                else:
                    
                    __knear(node.left,test,k,heap)
        heap=[]
        __knear(self.root,test,k,heap)      
        print(heap)
                



    def print(self):
        def print_h(Node):
            if (Node.is_leaf):
                print("leaf" , Node.point)
            else:
                print("mediana :" , Node.point ,"dimensao :",Node.dim)
               
                print_h(Node.left)
                print_h(Node.right)
          

        print_h(self.root)

tree= kdtree ([(2,3),(5,4),(9,6),(4,7),(8,1),(7,2)],2)
tree = kdtree ([(1,1),(2,2),(3,3),(4,4),(5,5)],2)

# a=[(100,100)]
# for i in range(50000):
#     a.append((random(),random()))



tree.knearest((8,2),5)