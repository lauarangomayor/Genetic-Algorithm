import random
bestTime = 0
INF = float("inf")

# Selects two individuals based on given probabilities of selection.
def selection(generation, probabilities):
    nGen = len(generation)
    posIndA= nGen-1
    probIndA = random.random()

    for pos in range(nGen-1):
        if probIndA == probabilities[pos] or (probIndA > probabilities[pos] and probIndA < probabilities[pos+1]):
            posIndA = pos
            break

    posIndB = posIndA
    while posIndA == posIndB:
        probIndB = random.random()
        for pos in range(nGen-1):
            if probIndB == probabilities[pos] or (probIndB > probabilities[pos] and probIndB < probabilities[pos+1]):
                posIndB = pos
                break

    return (generation[posIndA], generation[posIndB])

# Discards the worst individuals of the generation.
def discard(generation):
    nGen = len(generation)
    return (generation[:nGen//2])

# It creates two individuals based on two other individuals.
# note that we are not correcting if a machine can or not execute the operation.
def cross(indA, indB):
    # The cross is only going to take place between the second and the
    # antepenultimate element for ensure a change
    global machineOperationTime

    pivot = random.randint(1, len(machineOperationTime)-2)
    newIndividualA = indA[:pivot] + indB[pivot:]
    newIndividualB = indB[:pivot] + indA[pivot:]
    return (newIndividualA, newIndividualB)

def mutation(ind, prob):
    global machineOperationTime, bestTime
    p = random.randint(1,100)
    taken = {}
    if p < prob*100: 

        operationPosition = random.randint(0,len(machineOperationTime)-1)
        machine = random.randint(0,len(machineOperationTime[0])-1)

        while machineOperationTime[operationPosition][machine] == INF:
            machine = random.randint(0,len(machineOperationTime[0])-1)

        #startTime = random.randint(0,maxPossibleTime-1)
        startTime = 0 if machine not in taken else random.randint(0,bestTime)
        taken[machine] = True

        ind[operationPosition] = [machine, startTime]

    return ind

def findNextClosestToEndTime(ind, myEndTime, myOperation, myMachine):
    minimum = float('inf')
    operationMin = -1
    for operation in range(len(ind)):
        if operation == myOperation:
            continue
        machine = ind[operation][0]
        startTime = ind[operation][1]

        # If another operation uses the same machine and it is executed after my operation
        # has finished then we search the one that is executed earlier and penalize the wait time.
        if (machine == myMachine):
            if (startTime > myEndTime and startTime < minimum):
                minimum = startTime;
                operationMin = operation

    return (operationMin, minimum)

def fitness(ind):
    # Fitness function, the bigger result it returns the better is cualified de the individual.

    global tasks, machineOperationTime, maxPossibleTime
    fitness = 100;
    scale = fitness / (len(ind) * len(ind))
    for operationA in range(len(ind)):
        machineA, startTimeOfA = ind[operationA][0], ind[operationA][1]
        endTimeOfA = startTimeOfA + machineOperationTime[operationA][machineA]
        taskOfOperationA = tasks[operationA]

        # For each operation, find the next operation done in the same machine and penalyze their gaps.
        # The less time between them, the better.
        tup = findNextClosestToEndTime(ind, endTimeOfA, operationA, machineA)
        if (tup[0] != -1):
            fitness = fitness - ((tup[1] - endTimeOfA)*scale)

        # If after a cross of individuals an operation lands in a machine that cannot execute it then we penalize.
        if machineOperationTime[operationA][ind[operationA][0]] == INF:
            fitness = fitness - (scale + len(ind))

        for operationB in range(len(ind)):

            # If the operation evaluated is equal the one we are comparing it with, then skip.
            if operationA == operationB:  
                continue
            
            machineB, startTimeOfB = ind[operationB][0], ind[operationB][1]
            endTimeOfB = startTimeOfB + machineOperationTime[operationB][machineB]
            taskOfOperationB = tasks[operationB]

            # Ordering must be guaranteed.
            if (taskOfOperationA == taskOfOperationB):
                if (operationA < operationB and startTimeOfB <= endTimeOfA):
                    fitness = fitness - scale

                elif (operationA > operationB and startTimeOfA <= endTimeOfB):
                    fitness = fitness - scale

            # Overlapping must no be permited.
            if (machineA == machineB):
                if (startTimeOfA >= startTimeOfB and startTimeOfA <= endTimeOfB):
                    fitness = fitness - (scale + len(ind))

    return fitness


# Generates a single random individual.
def newIndividual():
    global machineOperationTime, bestTime
    ind = []

    # Dict to mark wich machine has been taken at the time zero.
    taken = {}

    # For each operation it creates a pair array where you'll find 
    # its machine m and its startTime t0, as [m, t0]
    for operation in range(len(machineOperationTime)):

        machine = random.randint(0,len(machineOperationTime[0])-1)

        while machineOperationTime[operation][machine] == INF:
            machine = random.randint(0,len(machineOperationTime[0])-1)

        startTime = 0 if machine not in taken else random.randint(0,bestTime)
        taken[machine] = True

        ind.append([machine, startTime])

    return ind

# Generates the first generations of rando individuals.
def firstGeneration(nIndGen):

    generation = []
    while len(generation) < nIndGen:
        generation.append(newIndividual())

    return generation


# Gives the best of the bests times if possible to finish the task set.
def bestTimeOfTaskSet():
    global tasks, machineOperationTime

    minTimeOperation = []
    i, j, minAc = 0, tasks[0], 0
    while i < len(machineOperationTime):
        
        if tasks[i] == j:
            # if the operation belongs to the task evaluated previously, then sums its time.
            minAc += min(machineOperationTime[i])
        else:
            # If the operation refers to a new task then add the minum time for the task and change the evaluated task.
            minTimeOperation.append(minAc)
            minAc, j = min(machineOperationTime[i]), tasks[i]

        i += 1
    return max(minTimeOperation)

def getProbability(generation):
    individualFitness, totalFitness = [], 0
    for ind in generation:
        indFitness = fitness(ind)
        if indFitness < 0:
            indFitness = 0
        individualFitness.append(indFitness)
        totalFitness += indFitness

    for fit in range(len(individualFitness)):
        individualFitness[fit] = individualFitness[fit]/totalFitness

    for i in range(1,len(individualFitness)):
        individualFitness[i] += individualFitness[i-1]
    return individualFitness


# Genetic function to create generations based on the best individual of the first generation.
def genetic(nIndGen,nGen,pMut):
    global machineOperationTime
    
    generation = firstGeneration(nIndGen)
    while nGen > 0:
        generation.sort(key = fitness, reverse = True)
        bestIndividualOfGen = generation[0]

        # Refers to the termination time of the best individual of the generation.
        taskSetFinishTime = -float('inf')
        for i in range(len(bestIndividualOfGen)):
            taskSetFinishTime = max(taskSetFinishTime, bestIndividualOfGen[i][1] + machineOperationTime[i][bestIndividualOfGen[i][0]])

        # Here we get the probability of each individual for being elected acord to its fitness.
        probabilities = getProbability(generation)

        generation = discard(generation)
        children = []
        while len(children) + len(generation) < nIndGen:
            parentA, parentB = selection(generation, probabilities)
            childA, childB = cross(parentA,parentB)
            childA = mutation(childA, pMut)
            childB = mutation(childB, pMut)
            children.append(childA)
            children.append(childB)
        generation = generation + children

        nGen = nGen - 1

    return taskSetFinishTime

# time for each task on every machine.
machineOperationTime = [[10,8,INF],
                        [INF,12,INF], 
                        [4, 6, 5],
                        [11, 18, INF],
                        [20, INF, INF],
                        [INF, 12, 16],
                        [7, 12, 4],
                        [14, 11, 9]]

# set of secuential opperations of each task
tasks = [0, 0, 0, 1, 1, 2, 2, 2]

bestTime = bestTimeOfTaskSet()

gen = genetic(200,200,0.1)
print(gen)
