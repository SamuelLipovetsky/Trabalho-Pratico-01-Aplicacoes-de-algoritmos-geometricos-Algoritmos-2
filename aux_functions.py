import random
#use the fact that classes are always the last attribute in a keel dat file
#Ignore information about attributes , returns a list of points
def keel_dat_reader(file_path,separator):
    with open(file_path,"r") as f:
        lines=f.readlines()
        offset=0
        points=[]
        for (index,i) in enumerate(lines):
            if i[0]=='@': 
                offset+=1
            else:
                #removing spaces and spliting line
                i = i.strip().split(separator)
                temp = i[-1]
                #remove the last element bc it can be a string 
                i =i[:-1]
                i= list(map(float,(i)))
                #inserting a tuple of (id,class)
                #in each line for the knn algorithm
                i.append((index-offset,temp))
                lines[index]= i
                
        return lines[offset:]

#return a train , test list 
#there is a fake shuffle using random
#bc some datasets have one class in the first half of elements
#and the other class in the second half of the elements
def split(list,pct):
    
    train=[]
    test =[]
    for i in list:
        rand_int = random.randrange(0,10,1)
        # print(rand_int, pct*10.0)
        if random.randrange(0,10,1)> (pct*10.0):
            test.append(i)
        else:
            train.append(i)
    # print(len(train),len(test))
    return (train, test)
