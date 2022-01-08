from xnn import *
import random
from timeit import default_timer as timer
from aux_functions import *

#test quickselects and generate data for plots
def test_quickselect():
    random.seed=10
    bf_time=[]
    qs_time=[]
    # testing quickselect method
    a= [ [random.randrange(1,100,1)] for i in range(100000)]
    b= [ [random.randrange(1,100,1)] for i in range(200000)]
    c= [ [random.randrange(1,100,1)] for i in range(300000)]
    d =[ [random.randrange(1,100,1)] for i in range(400000)]
    for Index,i in enumerate([a,b,c,d]):

        start=timer()
        bf = sorted(i,key=lambda x:x[0])[len(i)>>1][0]
        end=timer()
        bf_time.append(end - start)
    
        start=timer()
        qs = quick_select(i,len(i)>>1,0)[0] 
        end=timer()
      
        qs_time.append(end - start)
         
        assert qs==bf
   
    return bf_time,qs_time,[len(a),len(b),len(c),len(d)]


#test  knn and generate data for plots
def test_knn():
    #Testing knn method
    bf_time=[]
    knn_time=[]
    p_lists=[]
    p_lists.append(keel_dat_reader('./datasets/appendicitis.dat',","))
    p_lists.append(keel_dat_reader('./datasets/sonar.dat',", "))
    p_lists.append(keel_dat_reader('./datasets/haberman.dat',", "))
    p_lists.append(keel_dat_reader('./datasets/bupa.dat',", "))
    p_lists.append(keel_dat_reader('./datasets/monk-2.dat',", "))
    p_lists.append(keel_dat_reader('./datasets/banana.dat',","))
    p_lists.append(keel_dat_reader('./datasets/phoneme.dat',","))

    for i in p_lists:
        k=15
        #knn
        test=i[-1]
        tree = xnn(i)
        heap=[]
        start = timer()
        tree.knear(tree.root, test, k, heap)
        end = timer()
        knn_time.append(end-start)
        p1=sorted([-j.dist for j in heap])
        
        #brute force
        start = timer()
        dists = [euc(i[j][:-1],test[:-1]) for j in range(len(i))]
        p2 = sorted(dists)[:k]
        end = timer()
        bf_time.append(end - start)
          
        assert p2==p1
    return (knn_time,bf_time, 
    #these are the labels for the plots
    ["app. \n n = " + str(len(p_lists[0]))    ,
    "son \n n = " + str(len(p_lists[1]))   ,
    "hab\n n = " + str(len(p_lists[2]))    ,
     "bup \n n = " + str(len(p_lists[3]))    , 
    "mon \n n = " + str(len(p_lists[4]))    , 
    "ban \n n = " + str(len(p_lists[5]))    ,
    "phon \n n = " + str(len(p_lists[6]))   
      ])
    
