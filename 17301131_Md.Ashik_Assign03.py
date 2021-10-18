
# id 17301131
import math
import random

graph={}
c = [0,0]
def NumberOfNode(branchingFactor,depth):
    c = 0
    for i in range(0,depth+1):
        c = c+pow(branchingFactor,i)
    return c
m = 0

def minimax(position, depth, maximizingPlayer):
    if depth == 0:
        c[0] = c[0] + 1
        return graph[position][0]
    if maximizingPlayer:
        maxEval = -math.inf
        for i in graph[position]:
            eval = minimax(i,depth-1,False)
            maxEval = max(maxEval,eval)
        return maxEval
    else:
        minEval = math.inf
        for i in graph[position]:
            eval = minimax(i,depth-1,True)
            minEval = min(minEval,eval)
        return minEval
def m(position, depth,alpha, beta, maximizingPlayer):
    if depth == 0:
        c[1] = c[1] +1
        return graph[position][0]
    if maximizingPlayer:
        maxEval = -math.inf
        for i in graph[position]:
            eval = m(i,depth-1,alpha, beta,False)
            maxEval = max(maxEval,eval)
            alpha = max(alpha,eval)
            if beta<=alpha:
                break
        return maxEval
    else:
        minEval = math.inf
        for i in graph[position]:
            eval = m(i,depth-1,alpha,beta,True)
            minEval = min(minEval,eval)
            beta = min(beta,eval)
            if beta<=alpha:
                break
        return minEval
#a = [3,12,8,2,4,6,14,5,2]

def Create_Graph(b,d):
        k = 0
        c = 0
        for i in range(0, NumberOfNode(b,d)):
            if i not in graph.keys():
                graph[i] = list()
            for j in range(0,d+1):
                c = c +1
                if i>=(NumberOfNode(b,d)-pow(b,d)):
                    graph[i].append(random.randrange(minr,maxr+1))
                    #graph[i].append(a[k])
                    #k = k +1
                    break
                else:
                    graph[i].append(c)


minr = 0
maxr = 0

if __name__ == '__main__':
    d = int(input()) * 2
    b = int(input())
    r = input().strip().split()
    minr = int(r[0])
    maxr = int(r[1])
    Create_Graph(b,d)
    #print(graph)
    a1 = minimax(0, d, True)
    a2 = m(0, d,-math.inf, math.inf, True)
    print("Depth:",d)
    print("Branch:",b)
    print("Terminal States (Leaf Nodes):",pow(b,d))
    print("Maximum amount:",a1)
    print("Comparisons:",c[0])
    print("Comparisons:",c[1])

