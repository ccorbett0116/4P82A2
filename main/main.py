import os
import pickle
import time
import numpy as np
import math
import random
import operator

from noise import pnoise2

from Colour import Colour
from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import pngToNpArray as ptnp
import multiprocessing
import toInfix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

toolboxes = []
imageW, imageH = 64, 64
inputName = '../64x64.png'
inputData = ptnp.readImage(inputName)


def evalIMG(individual, inputData, pset):
    # Transform the tree expression in a callable function
    func = gp.compile(expr=individual, pset=pset)
    # Evaluate the mean squared error between the expression
    # and the real image
    error = 0
    for x in range(imageW):
        for y in range(imageH):
            # Multiplier to give more weight to the center of the image
            multiplier = (1 - abs((x / (imageW / 2)) - 1)) + (1 - abs((y / (imageH / 2)) - 1))
            try:
                # Calculate the error
                error += sum(((func((x / (imageW / 2)) - 1, (y / (imageH / 2)) - 1).getColour() - inputData[x][
                    y]) ** 2) * multiplier).astype(np.float64)
            except (ValueError, OverflowError, ZeroDivisionError):
                error += 999999999
    # Return the error rate
    return error / (imageW * imageH),


# Function to divide two numbers, with a special case for division by 0
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


# Function to log the value of x, with special cases for 0 and negative numbers
def protectedLog(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1 * math.log(abs(x))
    else:
        return math.log(x)


# Function to convert RGB values to a Colour object
def toColour(r, g, b):
    return Colour(r, g, b)


# Function to return the max of two numbers
def gt(a, b):
    if (a > b):
        return a
    else:
        return b

# Function to return the min of two numbers
def lt(a, b):
    if (a < b):
        return a
    else:
        return b

# Function to return the square of a number
def exp(a):
    return math.pow(a, 2)

# Function to return the noise value at a given point
def noise(x, y, octaves=5, persistence=0.45, lacunarity=2.0):
    return pnoise2((x/2)+0.5, (y/2)+0.5, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

# Function to return the noise value at a given point
def noise2(x, y, octaves, persistence, lacunarity):
    # clamp 0 < octaves <= 12
    # clamp 0 <= persistence <= 1
    # clamp 1 <= lacunarity <= 4
    octaves = max(1, min(round(octaves), 12))
    persistence = max(0, min(persistence, 1))
    lacunarity = max(1, min(lacunarity, 4))
    return pnoise2((x/2)+0.5, (y/2)+0.5, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # create a fitness class that minimizes the fitness value
creator.create("Individual", gp.PrimitiveTree,
               fitness=creator.FitnessMin)  # create an individual class that is a primitive tree with a fitness attribute

# Function to run the EA
def runEa(seed=7246325, popSize=700, mode="notSeeded", seedArr=[], migFreq=0, migQuant=0, numPops=1, pool=None, ngen=80, elitism=True, verbose=True):
    # Set the seed for the random number generator
    random.seed(seed)
    np.random.seed(seed)

    # Create the toolbox
    toolboxes = []
    # Populate the toolbox
    for i in range(numPops):
        toolboxes.append(base.Toolbox())
    ##RUN
    psets = []
    pops = []
    hofs = []
    cxpbs = []
    mutpbs = []
    # Add the necessary operators to the toolbox
    for i in range(numPops):
        psets.append(gp.PrimitiveSetTyped("main", [float, float], Colour))
        psets[i].addPrimitive(operator.add, [float, float], float)
        psets[i].addPrimitive(operator.sub, [float, float], float)
        psets[i].addPrimitive(operator.mul, [float, float], float)
        psets[i].addPrimitive(protectedDiv, [float, float], float)
        psets[i].addPrimitive(protectedLog, [float], float)
        psets[i].addPrimitive(toColour, [float, float, float], Colour)
        psets[i].addPrimitive(math.cos, [float], float)
        psets[i].addPrimitive(math.sin, [float], float)
        psets[i].addPrimitive(math.tan, [float], float)
        psets[i].addPrimitive(gt, [float, float], float)
        psets[i].addPrimitive(lt, [float, float], float)
        psets[i].addPrimitive(exp, [float], float)
        psets[i].addPrimitive(operator.abs, [float], float)
        psets[i].addPrimitive(operator.neg, [float], float)
        psets[i].addPrimitive(noise, [float, float], float)
        #psets[i].addPrimitive(noise2, [float, float, float, float, float], float)
        psets[i].addEphemeralConstant("rand101", partial(random.uniform, -1, 1), float)
        psets[i].renameArguments(ARG0='x')
        psets[i].renameArguments(ARG1='y')
        toolboxes[i].register("expr", gp.genFull, pset=psets[i], min_=5, max_=11)
        toolboxes[i].register("individual", tools.initIterate, creator.Individual, toolboxes[i].expr)
        toolboxes[i].register("population", tools.initRepeat, list, toolboxes[i].individual)
        toolboxes[i].register("compile", gp.compile, pset=psets[i])
        toolboxes[i].register("evaluate", evalIMG, inputData=inputData, pset=psets[i])
        toolboxes[i].register("select", tools.selTournament, tournsize=2)
        toolboxes[i].register("mate", gp.cxOnePoint)
        toolboxes[i].register("expr_mut", gp.genFull, min_=3, max_=9)
        toolboxes[i].register("mutate", gp.mutUniform, pset=psets[i], expr=toolboxes[i].expr_mut)
        toolboxes[i].decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=30))
        toolboxes[i].decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=30))
        toolboxes[i].register("migration", tools.migRing, k=migQuant, selection=tools.selBest, replacement=tools.selWorst)
        if pool is not None:
            toolboxes[i].register("map", pool.map)
        pops.append(toolboxes[i].population(n=popSize))
        # If the mode is seeded, seed the population with the best individuals from previous seeded runs
        if mode == "seeded":
            for j in range(len(seedArr)):
                pops[i][j] = seedArr[j]

        hofs.append(tools.HallOfFame(3))
        cxpbs.append(0.95)
        mutpbs.append(0.35)

    # Init stats
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("med", np.median)  # register the median function for the statistics1
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Run the EA
    pops, log = algorithms.eaSimple(pops, toolboxes, cxpbs=cxpbs, mutpbs=mutpbs, ngen=ngen, stats=mstats,
                                    halloffames=hofs, verbose=verbose, elitism=elitism, k=migFreq)
    return pops, hofs, mstats, toolboxes, log

# Main function
if __name__ == "__main__":
    # Set the seed
    random.seed(7246325)
    startTime = time.time()
    manager = multiprocessing.Manager()
    bests = []
    mins = []
    meds = []
    avgs = []
    unseededBest = float(99999999)
    tracker = -1
    toolbox = base.Toolbox()
    with multiprocessing.Pool(os.cpu_count()) as pool:
        try:
            print("Unseeded Runs:")
            order = []
            bests = []
            mins = []
            meds = []
            avgs = []
            unseededBest = float(99999999)
            tracker = -1
            unseededTracker = -1
            seededTracker = -1
            seededBest = float(99999999)
            # Run the EA 10 times
            for i in range(1, 11):
                print(f"Run {i}/10")
                # Data for the run
                pops, hofs, mstats, toolboxes, logs = runEa(pool=pool, seed=i, popSize=700, ngen=110, elitism=True)
                log = logs[0].chapters.get("fitness") 
                mins.append(log.select("min"))
                meds.append(log.select("med"))
                avgs.append(log.select("avg"))
                toolbox = toolboxes[0]
                bests.append(hofs[0][0])
                order.append(hofs[0][0].fitness.values[0])
                # Check if unseeded best and update the hofs
                if type(unseededBest) == float:
                    unseededBest = hofs[0][0]
                    unseededTracker = i
                # If the current best is better than the previous best, update the best
                elif hofs[0][0].fitness.values[0] < unseededBest.fitness.values[0]:
                    unseededBest = hofs[0][0]
                    unseededTracker = i
                best = hofs[0][0]
                func = toolbox.compile(expr=best)
                outputData = np.zeros((imageW, imageH, 3), dtype=np.uint8)
                # Compute output data for the image
                for x in range(imageW):
                    for y in range(imageH):
                        outputData[x][y] = func((x / (imageW / 2)) - 1, (y / (imageH / 2)) - 1).getColour()
                ptnp.saveImage(outputData, f'outputs/unseeded/thumbnailRun{i}.png')
                bigImageX = 2048
                bigImageY = 2048
                bigImage = np.zeros((bigImageX, bigImageY, 3), dtype=np.uint8)
                # Compute big image (larger resolution)
                for x in range(bigImageX):
                    for y in range(bigImageY):
                        bigImage[x][y] = func((x / (bigImageX / 2)) - 1, (y / (bigImageY / 2)) - 1).getColour()
                ptnp.saveImage(bigImage, f'outputs/unseeded/bigOutputRun{i}.png')
                print(toInfix.functional_to_infix(toInfix.parse_functional_expression(str(hofs[0][0]))))
                print(hofs[0][0].fitness.values[0])

            # More data collection
            avgMins = [np.mean([mins[j][i] for j in range(len(mins))]) for i in range(len(mins[0]))]
            avgMeds = [np.mean([meds[j][i] for j in range(len(meds))]) for i in range(len(meds[0]))]
            avgAvgs = [np.mean([avgs[j][i] for j in range(len(avgs))]) for i in range(len(avgs[0]))]
            plt.plot(avgMins, label="Unseeded Minimum")
            plt.ylabel("Minimum Fitness")
            plt.xlabel("Generation")
            plt.title("Minimum Fitness vs Generations")
            plt.legend()
            plt.savefig("outputs/unseeded/minFitness.png")
            plt.clf()
            plt.plot(avgMeds, label="Unseeded Median")
            plt.ylabel("Median Fitness")
            plt.xlabel("Generation")
            plt.title("Median Fitness vs Generations")
            plt.legend()
            plt.savefig("outputs/unseeded/medFitness.png")
            plt.clf()
            plt.plot(avgAvgs, label="Unseeded Average")
            plt.ylabel("Average Fitness")
            plt.xlabel("Generation")
            plt.title("Average Fitness vs Generations")
            plt.legend()
            plt.savefig("outputs/unseeded/avgFitness.png")
            plt.clf()
            print("Computing additional bests for seeding")
            for i in range(1, 11):
                print(f"Run {i}/10")
                pops, hofs, mstats, toolboxes, logs = runEa(pool=pool, seed=i+7240450, popSize=700, ngen=110, elitism=True, verbose=False)
                bests.append(hofs[0][0])
            mins = []
            meds = []
            avgs = []
            seededBests = []
            print("Seeded Runs:")
            # Run the EA 10 times
            for i in range(1, 11):
                print(f"Run {i}/10")
                random.shuffle(bests)
                seedArr = bests[:10]
                # Data for the run
                pops, hofs, mstats, toolboxes, logs = runEa(pool=pool, seed=7246325, popSize=700, ngen=110, mode="seeded", seedArr=seedArr, elitism=True)
                log = logs[0].chapters.get("fitness")
                mins.append(log.select("min"))
                meds.append(log.select("med"))
                avgs.append(log.select("avg"))
                toolbox = toolboxes[0]
                best = hofs[0][0]
                seededBests.append(best)
                order.append(hofs[0][0].fitness.values[0])
                # Check if seeded best and update the hofs
                if type(seededBest) == float:
                    seededBest = hofs[0][0]
                    seededTracker = i
                elif hofs[0][0].fitness.values[0] < seededBest.fitness.values[0]:
                    seededBest = hofs[0][0]
                    seededTracker = i
                func = toolbox.compile(expr=best)
                outputData = np.zeros((imageW, imageH, 3), dtype=np.uint8)
                # Compute output data for the image
                for x in range(imageW):
                    for y in range(imageH):
                        outputData[x][y] = func((x / (imageW / 2)) - 1, (y / (imageH / 2)) - 1).getColour()
                ptnp.saveImage(outputData, f'outputs/seeded/thumbnailRun{i}.png')
                bigImageX = 2048
                bigImageY = 2048
                bigImage = np.zeros((bigImageX, bigImageY, 3), dtype=np.uint8)
                # Compute big image (larger resolution)
                for x in range(bigImageX):
                    for y in range(bigImageY):
                        bigImage[x][y] = func((x / (bigImageX / 2)) - 1, (y / (bigImageY / 2)) - 1).getColour()
                ptnp.saveImage(bigImage, f'outputs/seeded/bigOutputRun{i}.png')

            print(toInfix.functional_to_infix(toInfix.parse_functional_expression(str(hofs[0][0]))))
            print(hofs[0][0].fitness.values[0])

            # More data collection
            avgMins = [np.mean([mins[j][i] for j in range(len(mins))]) for i in range(len(mins[0]))]
            avgMeds = [np.mean([meds[j][i] for j in range(len(meds))]) for i in range(len(meds[0]))]
            avgAvgs = [np.mean([avgs[j][i] for j in range(len(avgs))]) for i in range(len(avgs[0]))]
            plt.plot(avgMins, label="Seeded Minimum")
            plt.ylabel("Minimum Fitness")
            plt.xlabel("Generation")
            plt.title("Minimum Fitness vs Generations")
            plt.legend()
            plt.savefig("outputs/seeded/minFitness.png")
            plt.clf()
            plt.plot(avgMeds, label="Seeded Median")
            plt.ylabel("Median Fitness")
            plt.xlabel("Generation")
            plt.title("Median Fitness vs Generations")
            plt.legend()
            plt.savefig("outputs/seeded/medFitness.png")
            plt.clf()
            plt.plot(avgAvgs, label="Seeded Average")
            plt.ylabel("Average Fitness")
            plt.xlabel("Generation")
            plt.title("Average Fitness vs Generations")
            plt.legend()
            plt.savefig("outputs/seeded/avgFitness.png")
            plt.clf()
        except KeyboardInterrupt:
            print("Interrupted. Cleaning up...")
    with open('outputs/bests.pkl', 'wb') as f:
        pickle.dump(bests, f)
        pickle.dump(seededBests, f)
    print(f"Time elapsed: {time.time() - startTime} seconds")
    print("Unseeded best:")
    print(toInfix.functional_to_infix(toInfix.parse_functional_expression(str(unseededBest))) + " with fitness " + str(unseededBest.fitness.values[0]) + " at run " + str(unseededTracker))
    print("Seeded best:")
    print(toInfix.functional_to_infix(toInfix.parse_functional_expression(str(seededBest))) + " with fitness " + str(seededBest.fitness.values[0]) + " at run " + str(seededTracker))
    print("Order of bests:")
    print(order)
    print("Time elapsed: " + str(time.time() - startTime) + " seconds")