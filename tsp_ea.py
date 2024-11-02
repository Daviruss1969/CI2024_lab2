from itertools import combinations
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from dataclasses import dataclass
from tqdm.auto import tqdm

FILE = 'cities/italy.csv' # Modify this path to select the problem

CITIES = pd.read_csv(FILE, header=None, names=['name', 'lat', 'lon'])

SIZE = len(CITIES)

DIST_MATRIX = np.zeros((SIZE, SIZE))

POPULATION_SIZE=20

OFFSPRING_SIZE=10

MAX_GENERATIONS=500_000

@dataclass
class Individual:
    genome: list[int]
    fitness: float = None

# Compute the matrix of distances in km
for c1, c2 in combinations(CITIES.itertuples(), 2):
    DIST_MATRIX[c1.Index, c2.Index] = DIST_MATRIX[c2.Index, c1.Index] = geodesic(
        (c1.lat, c1.lon), (c2.lat, c2.lon)
    ).km

def cost(tsp: list) -> int:
    """Return the cost of a tsp solution"""
    tot_cost = 0
    for c1, c2 in zip(tsp, tsp[1:]):
        tot_cost += DIST_MATRIX[c1, c2]

    return tot_cost

def fitness(tsp: list) -> int:
    return -cost(tsp)

def valid(tsp: list) -> bool:
    """Return true if the given tsp is valid"""

    # Check if the solution is a cycle
    if tsp[0] != tsp[SIZE]:
        return False

    # Check that the vertex is visited once
    already_visited = list()
    for i in range(SIZE):
        city = tsp[i]

        if city in already_visited:
            return False

        already_visited.append(city)

    # Check that all vertex is visited
    if len(already_visited) != SIZE:
        return False

    return True

def tournament_selection(population: list[Individual], n: int) -> Individual:
    """Perform a tournament selection, picking n random fighters from a given population and return the best one"""
    # Pick random fighters
    tournament = np.random.randint(0, len(population), size=n)
    fighters: list[Individual] = [population[i] for i in tournament]

    # Select the best one
    winner = fighters[0]
    for fighter in fighters[1:]:
        if fighter.fitness > winner.fitness:
            winner = fighter

    return winner

def inver_over_crossover(p1: Individual, p2: Individual) -> Individual:
    """Perform inver over crossover between to given parents and return the child"""
    genome = p1.genome.copy()

    # Select one city in the first parent
    starting_city = p1.genome[np.random.randint(1, SIZE)]

    # Select the closest city of the starting_city
    next_city = p2.genome[(p2.genome.index(starting_city) + 1) % len(p2.genome)]

    # Reverse the order
    if next_city != genome[0]: # Ensure that we don't break the cycle
        idx1, idx2 = genome.index(starting_city), genome.index(next_city)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        genome[idx1+1:idx2+1] = reversed(genome[idx1+1:idx2+1])

    return Individual(genome, fitness(genome))

def scramble_mutation(tsp: Individual, n: int) -> Individual:
    """Tweak function that will select strength random indexes and shuffle them"""
    genome = tsp.genome.copy()

    # Just perform a swap if n is one or less
    if n <= 1:
        return Individual(swap(genome), fitness(genome))

    # Check boundaries
    if n > SIZE:
        n = SIZE

    # Select n unique indexes
    indexes = set(np.random.randint(1, SIZE, size=n))
    
    # Get the values of thoses selected indexes
    values_to_scramble = [genome[i] for i in indexes]

    np.random.shuffle(values_to_scramble)

    # Replace them in the solution
    for i, scrambled_value in zip(indexes, values_to_scramble):
        genome[i] = scrambled_value

    return Individual(genome, fitness(genome))

def swap(tsp: list) -> list:
    """Perform a random swap between two element in a list"""
    # Select two random indexes
    first_index = np.random.randint(1, SIZE) # Never update the first and last element to keep the cycle
    second_index = np.random.randint(1, SIZE)

    # Swap elements
    tmp = tsp[first_index]
    tsp[first_index] = tsp[second_index]
    tsp[second_index] = tmp
    
    return tsp

def init_individual() -> Individual:
    """Return a purely random individual for the TSP problem"""
    genome = list(range(SIZE))
    np.random.shuffle(genome)
    genome.append(genome[0]) # Add the first city and close the hamiltonian cycle
    return Individual(genome, fitness(genome))

def ea_tsp() -> Individual:
    """Function that performs the evolutionary algorithm"""
    # Initialization of the population
    population = [init_individual() for _ in range(POPULATION_SIZE)]

    # Constants for the 1/5 success rule
    n = 8
    i = 0
    success = 0
    
    # Generations
    for _ in tqdm(range(MAX_GENERATIONS)):
        offspring: list[Individual] = list()
        
        # Generate offspring
        for _ in range(OFFSPRING_SIZE):
            if np.random.random() < .5:
                # 1/5 success rule
                if i == 5:
                    if success > 1:
                        n += 1
                    elif success < 1 and n > 1:
                        n -= 1
                    success = 0
                    i = 0

                # perform mutation of one parent
                parent = tournament_selection(population, int(OFFSPRING_SIZE/3))
                new_individual = scramble_mutation(parent, n)
                i+= 1

                # 1/5 success rule
                if new_individual.fitness > parent.fitness:
                    success += 1
            else:
                # Take two parents and perform crossover
                parent1 = tournament_selection(population, int(OFFSPRING_SIZE/3))
                parent2 = tournament_selection(population, int(OFFSPRING_SIZE/3))
                new_individual = inver_over_crossover(parent1, parent2)

            offspring.append(new_individual)

        # Steady state -> add offspring to population and perform the survivor selection
        population.extend(offspring)
        population.sort(key=lambda i: i.fitness, reverse=True)
        population = population[:POPULATION_SIZE]

    return population[0]

# Perform algorightm
tsp = ea_tsp()

# Results
print(f"The solution is {'not ' if not valid(tsp.genome) else ''}valid, total cost : {cost(tsp=tsp.genome)}")
