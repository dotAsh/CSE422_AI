########TAST 1 To 13 -all is here ###########id: 17301131###########
import numpy as np
import random

def f(a): #this function returns Number of non-attacking pairs of board only
    TotalAttacking = 0
    for i in range(0,len(a)):
        k = i +1
        c = 1
        while k!=len(a):
            if ((a[i]==a[k])):
                TotalAttacking = TotalAttacking + 1
            if ((a[i]+c == a[k])):
                TotalAttacking = TotalAttacking + 1
            if ((a[i]-c == a[k])):
                TotalAttacking = TotalAttacking + 1
            c = c+1
            k +=1

    return compute_goal_fit(8)-TotalAttacking
def compute_goal_fit(n):  #returns 28 non-attacking pairs of goal state in 8 queen
    goal_fit = 0
    for i in range(n):
        goal_fit += i
    return goal_fit
def arr(x ):  # this function just display the 2d array or the a single chess board and queen positions
    a = np.arange(64).reshape(8,8)
    for c in range(0,len(x)):
            for l in range(0,len(x)):
                a[c,l]= 0

    for c in range(0,len(x)):
            a[7-x[c],c]= x[c]
    print(a)
    del(a)


def fitness(population):  #This function returns the array which contains the fitness of boards of the whole population
    arr = [0]*len(population)
    i =0
    sum = 0
    for a in population:
        arr[i] = f(a)
        sum = sum + arr[i]
        i = i +1
    i = 0
    for a in population:
        arr[i] = arr[i]/sum
        i = i +1
    return arr


def crossOver(x,y):
    j = int(len(population)/2)
    a = []
    for i in range(0,j):
        a.append(y[i])

    for i in range(j,len(y)):
        a.append(x[i])
    return a

def mutate(child):
    j = random.randrange(0,8)
    child[j]=random.randrange(0,8)
    return child

def select():
    a = []
    if (start_population == 10):
        a = [0,1,2,3,4,5,6,7,8,9]
    if (start_population == 4):
        a = [0,1,2,3]
    f =  fitness(population)
    return int(np.random.choice(a, 1, replace=True, p=f))

def GA(population, n, mutation_threshold = 0.3):
    nmax = 1000
    n = nmax
    peyeGechi = False
    print("please wait !!!!!!!!!!!!!! algo is running")
    while n > 0:
        new_population = []
        ff =  fitness(population)
        i = len(population)
        while i > 0:
            a = select()
            b = select()
            a2 = population[a]
            b2 = population[b]
            child = crossOver(a2,b2)
            if f(child) == 28:
                print("...done. \n")
                peyeGechi = True
                print("goal ", child , "found in ", nmax -n ,"generation! \n")
                return None
            if random.uniform(0,1) < 1.0:
                child = mutate(child)
            if f(child) == 28:
                print("...peyechi. \n")
                peyeGechi = True
                print("goal", child , "found in ", nmax -n ,"generation!\n")
                return None
            if f(child)>25 or 11>f(child)<15:
                new_population.append(child)
                i = i -1
        population = new_population
        n -=1
    if not peyeGechi:
        print("\nno goal in ", nmax,"generations!!\n")
    return None



if __name__ == '__main__':
    n = 8
    mutation_threshold = 0.3
    population = np.random.randint(8, size=(10, 8))
    start_population = len(population)
    print("This the population\n",population)
    print("This is the fitness values of the whole population!\n",fitness(population))
    """for i in population:
        arr(i)
    crossOver([2,4,7,4,8,5,5,2],[3,2,7,5,2,4,1,1])
    mutate([2,4,7,4,8,5,5,2])
    print(select())"""
    GA(population, n, mutation_threshold)







