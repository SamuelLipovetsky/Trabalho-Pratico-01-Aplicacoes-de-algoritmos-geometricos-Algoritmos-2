from xnn import *
import random
import time
#testing quickselect method
a=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]] #median=6
b=[[10],[9],[8],[7],[6],[5],[4],[3],[2],[1]] #mediam =5
c=[[3],[4],[5],[10],[13],[1],[2],[12],[10],[10],[10]]  #mediam =10
d =[ [i] for i in range(1000)]
random.shuffle(d)
print("Testing quickselect")
assert quick_select(a,len(a)>>1,0)[0] == 6
print("test 1 passed")
assert quick_select(b,len(b)>>1,0)[0] == 6
print("test 2 passed")
assert quick_select(c,len(c)>>1,0)[0] == 10
print("test 3 passed")
assert quick_select(d,len(d)>>1,0)[0] == 500
print("test 4 passed")

#Testing knn method
points=[ (random.sample(range(0, 10000), 10000)) for i in range(1000)]
for i in points:
    i.append( (0, random.randrange(0,1,1)))

#using kd tree and knn
k=10
test =random.sample(range(0, 10000), 10000)

test.append((100001,0))

tree = xnn(points)
heap=[]
start = time.time()
tree.knear(tree.root, test, k, heap)
end = time.time()
print("knn done in :" + str(end - start))
p1 = sorted([-i.dist for i in heap])

#bruteforce method
start = time.time()
dists = [euc(points[i][:-1],test[:-1]) for i in range(len(points))]
p2 = sorted(dists)[:k]
end = time.time()
print("bruteforce done in :"+str(end - start))



assert(p2==p1)
#using brute force method to calculate all distances



    