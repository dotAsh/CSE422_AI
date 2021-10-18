#P01 ID: 17301131
with open('input','r') as file:
    file = file.readlines()
    EE = int(file[1].strip().split()[0])
    NN = int(file[0].strip().split()[0])
    GOAL = int(file[2+EE].strip().split()[0])
graph={}
Senior = [0] * NN

def Create_Graph():
        for i in range(2, 2 + EE , 1):
            EDGE = file[i].strip().split()
            UU = int(EDGE[0])
            VV = int(EDGE[1])
            if UU not in graph.keys():
                graph[UU] = list()
            graph[UU].append(VV )
            #bidirectional
            if VV  not in graph.keys():
                graph[VV ] = list()
            graph[VV ].append(UU)

def BFS(SS):
    ChenaNodes = [False] * NN
    QQQ = []
    QQQ.append(SS)
    ChenaNodes[SS] = True
    while QQQ:
        SS = QQQ.pop(0)
        for i in graph[SS]:
            if ChenaNodes[i] == False:
                QQQ.append(i)
                ChenaNodes[i] = True
                Senior[i] = SS

def DisTance(SS, GOAL):
    if (GOAL == SS):
        return 0
    else:
        return 1 + DisTance(SS, Senior[GOAL]);

if __name__ == '__main__':
    Create_Graph()
    BFS(0)
    print(DisTance(0, GOAL))





#p02
with open('input','r') as file:
    file = file.readlines()
    EE = int(file[1].strip().split()[0])
    NN = int(file[0].strip().split()[0])
    GOAL = int(file[2+EE].strip().split()[0])
    NORAA = int(file[2+EE+1].strip().split()[0])
    LARAA = int(file[2+EE+2].strip().split()[0])
graph={}
Senior = [0] * NN

def Create_Graph():
        for i in range(2, 2 + EE , 1):
            EDGE = file[i].strip().split()
            UU = int(EDGE[0])
            VV = int(EDGE[1])
            if UU not in graph.keys():
                graph[UU] = list()
            graph[UU].append(VV )
            #bidirectional
            if VV  not in graph.keys():
                graph[VV ] = list()
            graph[VV ].append(UU)

def BFS(SS):
    ChenaNodes = [False] * NN
    QQQ = []
    QQQ.append(SS)
    ChenaNodes[SS] = True
    while QQQ:
        SS = QQQ.pop(0)
        for i in graph[SS]:
            if ChenaNodes[i] == False:
                QQQ.append(i)
                ChenaNodes[i] = True
                Senior[i] = SS

def DisTance(SS, GOAL):
    if (GOAL == SS):
        return 0
    else:
        return 1 + DisTance(SS, Senior[GOAL]);

if __name__ == '__main__':
    Create_Graph()
    BFS(NORAA)
    d1 = DisTance(NORAA,GOAL)
    BFS(LARAA)
    d2 = DisTance(LARAA,GOAL)
    if d1<d2:
        print("Nora")
    else:
        print("Lara")




#p03
import math
with open('input','r') as file:
    file = file.readlines()
    EE = int(file[1].strip().split()[0])
    NN = int(file[0].strip().split()[0])

graph={}
Senior = [0] * NN

def Create_Graph():
        for i in range(2, 2 + EE , 1):
            EDGE = file[i].strip().split()
            UU = int(EDGE[0])
            VV = int(EDGE[1])
            if UU not in graph.keys():
                graph[UU] = list()
            #graph[UU].append(VV )
            #Unidirectional
            if VV  not in graph.keys():
                graph[VV ] = list()
            graph[VV ].append(UU)
        

def BFS(SS):
    ChenaNodes = [False] * NN
    QQQ = []
    QQQ.append(SS)
    ChenaNodes[SS] = True
    while QQQ:
        SS = QQQ.pop(0)
        for i in graph[SS]:
            if ChenaNodes[i] == False:
                QQQ.append(i)
                ChenaNodes[i] = True
                Senior[i] = SS

def DisTance(SS, GOAL):
    if (GOAL == SS):
        return 0
    else:
        return 1 + DisTance(SS, Senior[GOAL]);

if __name__ == '__main__':
    Create_Graph()
    BFS( int(file[2+EE].strip().split()[0]))
    min = math.inf
    for i in range(0,  int(file[2+EE+1].strip().split()[0]) , 1):
        a = DisTance(  int(file[2+EE].strip().split()[0]) , int(file[2+EE+2+i].strip().split()[0]))
        if a < min:
            min = a
    print(min)




