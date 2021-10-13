##### python3

import numpy as np
from numpy.linalg import multi_dot
from sys import maxsize
import random

def RevisedSimplex(A, b, x, sf_multiplier, non_negativity=True):
    
    allowed_error = 1e-16
    
    # check feasibility of the origin
    slack_surplus = []
    for i in range(len(b)):
        # if RHS is negative, multiply the row by -1. labet the row as "surplus", otherwise label it as "slack"
        if b[i] < 0:
            slack_surplus.append("surplus")
            b[i] *= -1
            for j in range(len(A[i])):
                A[i][j] *= -1
        else:
            slack_surplus.append("slack")
                
    if non_negativity:
        for i in range(len(x)):
            slack_surplus.append("surplus")
            b.append(0)
            newRow = [0 for i in range(len(x))]
            newRow[i] = 1
            A.append(newRow)
    print("A:")
    print(A)
    
    print("###############")
    print(len(A))
    print(len(slack_surplus))
    print(slack_surplus)
    print("###############")
    
    first_b = [[b[i]] for i in range(len(b))]
    b = first_b.copy()
    print(b)
        
    # add slack variables and initialize xB
    xB = []
    slack = []
    surplus = []
    artificial = []
    constraintId = []
    index = len(x)
    for i in range(len(A)):
        if slack_surplus[i] == "slack":
            slack.append(index)
            xB.append(slack[len(slack)-1])
        elif slack_surplus[i] == "surplus":
            surplus.append(index)
            index += 1
            artificial.append(index)
            constraintId.append(i)
            xB.append(artificial[len(artificial)-1])
        index += 1
    artificial_counter = len(artificial)
    print("artificial variables:")
    print(artificial)
    
    
    # initialize cB
    cB = [(xB[i], 0) for i in range(len(xB))]
    
    # initialize B and B-inverse
    B = np.identity(len(xB))
    B_inverse = B.copy()
    print(B_inverse)
    
    # initialize xN
    xN = [i for i in range(len(x))]
    for i in surplus:
        xN.append(i)
    
    # initialize N
    N = A
    index = 0
    if constraintId != []:
        for i in range(len(N)):
            row = [0 for j in range(len(surplus))]
            if constraintId[index] == i:
                row[index] = -1
                index += 1
            N[i].extend(row)
    print("**************")
    print(N)
    print("**************")
    
    # initialize cN
    print("constraintId:", constraintId)
    if constraintId == []:
        cN = [(xN[i], x[i]) for i in range(len(x))]
    else:
        cN = [(i, 0) for i in range(len(x))]
        for i in surplus:
            cN.append((i, 0))
        print("first cN")
        print(cN)
        z = 0
        for i in constraintId:
            z -= b[i][0]
            for j in range(len(N[i])):
                variable = cN[j][0]
                value = cN[j][1]
                cN[j] = (variable, value + N[i][j])
                #cN[j][1] += A[i][j]
        print("second cN")
        print(cN)
    
    # PHASE I
    print("PHASE I ITERATIONS")
    if constraintId != []:
        while True:
            print("xB:", xB, end="                ")
            print("xN:", xN)
            print("cB:", cB, end= " ")
            print("cN:", cN)
            print("B:")
            for i in range(len(B)):
                print(B[i])
            print("B-1:")
            for i in range(len(B_inverse)):
                print(B_inverse[i])
            print("N:")
            for i in range(len(N)):
                print(N[i])
            print("b:", b)
        
            # (i) optimality check and identification of entering variable
            cBcoefs = [cB[i][1] for i in range(len(cB))]
            cNcoefs = [cN[i][1] for i in range(len(cN))]
        
            opt = multi_dot([cBcoefs, B_inverse, N]) - cNcoefs
            print("opt. check:", opt)
        
            tmp = 0
            enteringVarIndex = -1
            for i in range(len(opt)):
                if opt[i] < tmp:
                    tmp = opt[i]
                    enteringVarIndex = i
            if tmp >= 0 and artificial_counter > 0:
                
                cBcoefs = [cB[i][1] for i in range(len(cB))]
                w = multi_dot([cBcoefs, B_inverse, first_b])
                print("w1:",w)
                w += z
                print("z:",z)
                print("w2:",w)
                
                print("No solution")
                return
                
            enteringVar = xN[enteringVarIndex]
            print("entering variable:", enteringVar)
            if enteringVar in artificial:
                artificial_counter += 1
            print("artificial counter:", artificial_counter)
            
            # (ii) identify leaving variable
            b = np.matmul(B_inverse, first_b)
            col = [[N[i][enteringVarIndex]] for i in range(len(N))]
            col = np.matmul(B_inverse, col)
        
            tmp = maxsize
            leavingVarIndices_artificial = []
            leavingVarIndices_nonartificial = []
            #leavingVarIndex = -1
            
            print("_________identify leaving variable_______")
            print("b:", b)
            print("col:", col)
            for i in range(len(b)):
                if col[i][0] == 0:
                    continue
                if b[i][0] / col[i][0] >= 0 and col[i][0] > 0 and b[i][0] / col[i][0] <= tmp:
                    tmp = b[i][0] / col[i][0]
                    if xB[i] in artificial:
                        leavingVarIndices_artificial.append((i, tmp))
                    else:
                        leavingVarIndices_nonartificial.append((i, tmp))
            if leavingVarIndices_artificial != []:
                i = len(leavingVarIndices_artificial)-1
                minRatios = [leavingVarIndices_artificial[i][0]]
                while i > 0 and leavingVarIndices_artificial[i][1] == leavingVarIndices_artificial[i-1][1]:
                    minRatios.append(leavingVarIndices_artificial[i-1][0])
                    i -= 1
            else:
                i = len(leavingVarIndices_nonartificial)-1
                minRatios = [leavingVarIndices_nonartificial[i][0]]
                while i > 0 and leavingVarIndices_nonartificial[i][1] == leavingVarIndices_nonartificial[i-1][1]:
                    minRatios.append(leavingVarIndices_nonartificial[i-1][0])
                    i -= 1
                
            #print("leavingVarIndices:",leavingVarIndices)
            #print("length of minRatios:", len(minRatios))
            print("minRatios:", minRatios)
            i = random.randint(0,len(minRatios)-1)
            print(i)
            randomMinRatio = minRatios[i]
            leavingVarIndex = randomMinRatio
            
            # deals with degeneracy            
                        
                
            print("leaving var index:", leavingVarIndex)    
            leavingVar = xB[leavingVarIndex]
            print("leaving variable: ",leavingVar)
            #print(col)
            if leavingVar in artificial:
                artificial_counter -= 1
            print("artificial counter:", artificial_counter)
            
            print("-----------------")
            
            if tmp == maxsize:
                print("Infinity")
                return
        
            # (iii) update xB, cN, B, B_inverse, xN, cN, N
            xB[leavingVarIndex], xN[enteringVarIndex] = xN[enteringVarIndex], xB[leavingVarIndex]
            cB[leavingVarIndex], cN[enteringVarIndex] = cN[enteringVarIndex], cB[leavingVarIndex]
            for i in range(len(B)):
                B[i][leavingVarIndex], N[i][enteringVarIndex] = N[i][enteringVarIndex], B[i][leavingVarIndex]
            B_inverse = np.linalg.inv(B)
            
            # (iv) check if all artificial variables are dropped
            if artificial_counter <= 0:
                
                print("xB:", xB, end="                ")
                print("xN:", xN)
                print("cB:", cB, end= " ")
                print("cN:", cN)
                print("B:")
                for i in range(len(B)):
                    print(B[i])
                print("B-1:")
                for i in range(len(B_inverse)):
                    print(B_inverse[i])
                print("N:")
                for i in range(len(N)):
                    print(N[i])
                print("b:", b)
                
                print("-------final calculation------")
                
                cBcoefs = [cB[i][1] for i in range(len(cB))]
                cNcoefs = [cN[i][1] for i in range(len(cN))]
        
                opt = multi_dot([cBcoefs, B_inverse, N]) - cNcoefs
                print("opt. check:", opt)
        
                
                # if yes, check z - w
                cBcoefs = [cB[i][1] for i in range(len(cB))]
                w = multi_dot([cBcoefs, B_inverse, first_b])
                print("w1:",w)
                w += z
                #multi_dot([cBcoefs, B_inverse, N])
                
                #b_last = np.matmul(B_inverse, first_b)
                #w = np.matmul(cBcoefs, b_last) + z
                #print("last b:",b_last)
                print("z:",z)
                print("w2:",w)
                if abs(w) > allowed_error * sf_multiplier:
                    print("No solution")
                    return
                
                artificialPositions = []
                for i in range(len(xN)):
                    if xN[i] in artificial:
                        artificialPositions.append(i)
                xN_new = []
                cN_new = []
                N_new = [[] for i in range(len(N))]
                for j in range(len(xN)):
                    if not j in artificialPositions:
                        xN_new.append(xN[j])
                        if cN[j][0] < len(x):
                            cN[j] = (cN[j][0], x[cN[j][0]])
                        else:
                            cN[j] = (cN[j][0], 0)
                        cN_new.append(cN[j])
                        for i in range(len(N)):
                            N_new[i].append(N[i][j])
                xN = xN_new.copy()
                cN = cN_new.copy()
                #N = N_new.copy()
                print(N_new)
                N = np.matmul(B_inverse, N_new)
                print("N")
                print(N)
                
                for j in range(len(cB)):
                    if cB[j][0] < len(x):
                        cB[j] = (cB[j][0], x[cB[j][0]])
                    else:
                        cB[j] = (cB[j][0], 0)
                
                ERO = []
                for j in range(len(cB)):
                    ERO.append(cB[j][1])
                print("ERO:")
                print(ERO)
                for j in range(len(cN)):
                    nonbasicVarId = cN[j][0]
                    nonbasicVarVal = cN[j][1]
                    for i in range(len(ERO)):
                        #print(cN[j], "-(",ERO[i],"*",N[i][j],")=",end=" ")
                        nonbasicVarVal -= ERO[i]*N[i][j]
                        #print(cN[j])
                    cN[j] = (nonbasicVarId, nonbasicVarVal)
                
                b = np.matmul(B_inverse, first_b)
                first_b = b.copy()
                B = np.identity(len(xB))
                B_inverse = B.copy() 
                for i in range(len(cB)):
                    basicVarId = cB[i][0]
                    cB[i] = (basicVarId, 0)
                break
                
    
    # iterations
    print("PHASE II ITERATIONS")
    while True:
        
        print()
        print("xB:", xB, end="                ")
        print("xN:", xN)
        print("cB:", cB, end= " ")
        print("cN:", cN)
        print("B:")
        for i in range(len(B)):
            print(B[i])
        print("B-1:")
        for i in range(len(B_inverse)):
            print(B_inverse[i])
        print("N:")
        for i in range(len(N)):
            print(N[i])
        print("b:", b)
        
        # (i) optimality check and identification of entering variable
        cBcoefs = [cB[i][1] for i in range(len(cB))]
        cNcoefs = [cN[i][1] for i in range(len(cN))]
        
        opt = multi_dot([cBcoefs, B_inverse, N]) - cNcoefs
        print("opt. check:", opt)
        
        tmp = 0
        enteringVarIndex = -1
        for i in range(len(opt)):
            if opt[i] < tmp:
                tmp = opt[i]
                enteringVarIndex = i
        enteringVar = xN[enteringVarIndex]
        print("entering variable:", enteringVar)
        
        if tmp >= 0:
            b = np.matmul(B_inverse, first_b)
            solution = [(i, 0) for i in range(len(x)+len(slack)+len(surplus)+len(artificial))]
            print("===========")
            print("last b:")
            print(b)
            print("solution:")
            print(solution)
            
            for i in range(len(xB)):
                basicVarIndex = xB[i]
                #solution[basicVarIndex] = (cB[i][0], b[i][0])
                solution[basicVarIndex] = (basicVarIndex, b[i][0])
            print(solution)
            
            print("Bounded solution")
            for i in range(len(x)):
                print("{:.16f}".format(solution[i][1]), end=" ")
            return
        
        # (ii) identify leaving variable
        b = np.matmul(B_inverse, first_b)
        col = [[N[i][enteringVarIndex]] for i in range(len(N))]
        col = np.matmul(B_inverse, col)
        
        tmp = maxsize
        leavingVarIndex = -1
        for i in range(len(b)):
            if col[i][0] == 0:
                continue
            #if b[i][0] / col[i][0] >= 0 and col[i][0] > 0 and b[i][0] / col[i][0] < tmp:
            if col[i][0] > 0 and b[i][0] / col[i][0] < tmp:
                tmp = b[i][0] / col[i][0]
                leavingVarIndex = i
        leavingVar = xB[leavingVarIndex]
        print("leaving variable: ",leavingVar)
        print("-----------------")
        if tmp == maxsize:
            print("Infinity")
            return
        
        # (iii) update xB, cN, B, B_inverse, xN, cN, N
        xB[leavingVarIndex], xN[enteringVarIndex] = xN[enteringVarIndex], xB[leavingVarIndex]
        cB[leavingVarIndex], cN[enteringVarIndex] = cN[enteringVarIndex], cB[leavingVarIndex]
        for i in range(len(B)):
            B[i][leavingVarIndex], N[i][enteringVarIndex] = N[i][enteringVarIndex], B[i][leavingVarIndex]
        B_inverse = np.linalg.inv(B)
        
        
sf_multiplier = 1e2       
n, m = map(int, input().split()[:2])
A = []
for i in range(n):
    row = list(map(float, input().split()[:m]))
    for j in range(m):
        row[j] *= sf_multiplier
    A.append(row)
b = list(map(float, input().split()[:n]))
for i in range(n):
    b[i] *= sf_multiplier

x = list(map(float, input().split()[:m]))
RevisedSimplex(A, b, x, sf_multiplier)
