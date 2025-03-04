from itertools import product
from functools import reduce
import re

def targets(v):
    i = len(v) - 1
    if len(v) == 2:
        return [0]
    if v[i] < i:
        ret = [v[i]] + targets(v[:-1])
        #print('step ' + str(i) + ": " + str(ret))
        return ret
    else:
        waits = v[i] - i + 1 
        temp = targets(v[:-1])
        #print("temp:" + str(temp))
        target = temp[waits-1]
        #print("step " + str(i) + ": " + str(temp[0:waits-1] + [target] + temp[waits-1:]))
        return (temp[0:waits-1] + [target] + temp[waits-1:])

def joiners(v):
    i = len(v) - 1 
    if len(v) == 2:
        return [1]
    if v[i] < i:
        return [i] + joiners(v[:-1])
    else:
        waits = v[i] - i + 1
        temp = joiners(v[:-1])
        return (temp[0 : waits] + [i] + temp[waits:])
       
def naivetovec(vnaiv): 
    seenJoiners = []
    seenTargets = []
    v = [0]*(len(vnaiv) + 1)
    for j,t in vnaiv:
        if t in seenTargets:
            waits = len(list(filter(lambda x: x < j, seenJoiners)))
            if waits == 0:
                v[j] = t
            else:
                v[j] = waits + j - 1
                #print(j)
        else:
            v[j] = t
            seenTargets += [t]
          #  print(str(v))
        seenJoiners += [j]
    return v

def vecToNaive(v):
    j = joiners(v)
    t = targets(v)
    return list(zip(j,t))

def naiveToNewick(n):
    if len(n) < 1:
        raise "help"
    newick = None
    for i in range(len(n) - 1, -1, -1):
        if i == len(n) - 1:
            newick = str(n[i])
        else:
            newick = re.sub(rf"\b{str(n[i][1])}\b", str(n[i]), newick)
    return newick.replace(" ", "")

def vecToNewick(v):
    return naiveToNewick(vecToNaive(v))

def newickToNaiveHelper(s):
    ls = list(map(eval, re.findall(r'(\(\d+,\d+\))', s)))
    if len(ls) == 0:
        return []
    ls = sorted(ls, key = lambda x: -max(x[0], x[1]))
    join_s = str(ls[0]).replace(" ", "")
    join = sorted(ls[0], key = lambda x: -x)
    s_new = s.replace(join_s, str(join[1]))
    return [tuple(join)] + newickToNaiveHelper(s_new)

def newickToNaive(s):
    return newickToNaiveHelper(s)

# arr = [0, 0, 1, 3, 0]
# print(vecToNaive(arr))
# nai = [(4,0),(2,1),(3,1),(1,0)]
# print(naivetovec(nai))
# print(naiveToNewick(nai))



# def looprev(v):
#     nai = list(zip(joiners(v),targets(v)))
#     return naivetovec(nai) == v

# def loop(nai):
#     tmp = naivetovec(nai)
#     return list(zip(joiners(tmp),targets(tmp))) == nai
# #print("l " + str(looprev(arr)))
        
# #print(targets(arr))
# #print(joiners(arr))

def sequence(lists):
    if not lists:
        return [[]]
    else:
        return [x + [y] for x, y in product(sequence(lists[:-1]), lists[-1])]
    
# n = 4
# ls = [list(range(2*(i-1) + 1)) for i in list(range(1, n))]

def phylovecdomain(n) :
    ls = [list(range(2*(i-1) + 1)) for i in list(range(1, n))]
    return sequence([[0]] + ls)



# #result = [[0,0,x,y,z,w,u,v] for x in range(3) for y in range(5) for z in range(7) for w in range(9) for u in range(11) for v in range(13)  ]
# for n in range(2, 15):
#     print(n)
#     domain = phylovecdomain(n)

#     dblfact = reduce(int.__mul__, range(2*(n) - 3, 0, -2))
#     print("factcheck:" + str(len(domain) == dblfact))
#     bool = True
#     for ls in domain:
#         bool = bool and looprev(ls)
#         # if looprev(ls) == False:
#         #     tmp = ls
#         #     print(str(ls) + ", " + str(list(zip(joiners(tmp),targets(tmp)))) + ", " + str(naivetovec(list(zip(joiners(tmp),targets(tmp))))))
#     print("Did it work: " + str(bool))