from itertools import combinations
import pandas as pd
import numpy as np
from geopy.distance import geodesic

FILE = 'cities/italy.csv' # Modify this path to select the problem

MAX_STEPS = 500_000

CITIES = pd.read_csv(FILE, header=None, names=['name', 'lat', 'lon'])

SIZE = len(CITIES)

DIST_MATRIX = np.zeros((SIZE, SIZE))

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

def greedy_tsp(dist_matrix) -> list :
    """Function that solves a tsp problem on a given matrix of distances using a greedy approach"""
    dist = dist_matrix.copy()
    city = 0
    tsp = list()

    while not np.all(dist == np.inf):
        tsp.append(city)
        dist[:, city] = np.inf # don't select this city anymore
        closest = np.argmin(dist[city]) # select the closest city
        city = closest
    
    tsp.append(tsp[0]) # Add the first city and close the hamiltonian cycle
    return tsp

tsp = greedy_tsp(DIST_MATRIX)

print(f"The solution is {'not ' if not valid(tsp) else ''}valid, total cost : {cost(tsp=tsp)}")