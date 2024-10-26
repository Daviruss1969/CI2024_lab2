from itertools import combinations
import pandas as pd
import numpy as np
from icecream import ic
from geopy.distance import geodesic
CITIES = pd.read_csv('cities/vanuatu.csv', header=None, names=['name', 'lat', 'lon'])

DIST_MATRIX = np.zeros((len(CITIES), len(CITIES)))
for c1, c2 in combinations(CITIES.itertuples(), 2):
    DIST_MATRIX[c1.Index, c2.Index] = DIST_MATRIX[c2.Index, c1.Index] = geodesic(
        (c1.lat, c1.lon), (c2.lat, c2.lon)
    ).km

def tsp_cost(tsp):
    # assert tsp[0] == tsp[-1]
    # assert set(tsp) == set(range(len(CITIES))) # Bad code

    tot_cost=0
    for c1, c2 in zip(tsp, tsp[1:]):
        tot_cost += DIST_MATRIX[c1, c2]
    return tot_cost


city_names = np.array([c['name'] for _, c in CITIES.iterrows()])

# First greedy algorithm
dist = DIST_MATRIX.copy()
city = 0
tsp = list()
while not np.all(dist == np.inf):
    tsp.append(city)
    dist[:, city] = np.inf # bad code apparement
    closest = np.argmin(dist[city])
    print(f"{city_names[city]} -> {city_names[closest]}")
    city = closest

ic(tsp)
ic(tsp_cost(tsp))

## Second greedy algorithm (je le laisse coder)
# segments = [([({c1, c2}, DIST_MATRIX[c1, c2]) for c1, c2 in combinations(CITIES.itertuples(), 2)])]