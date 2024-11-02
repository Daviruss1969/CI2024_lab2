# Travelling salesman problem

This repository contains my work for the Lab 2 of the computational intelligence courses on the [Travelling salesman problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem).

## Notebook

The [tsp.ipnyb](./tsp.ipynb) file contains my reflexion about the problem. How I end up step by step with this solution.

## Code

The [tsp_greedy.py](./tsp_greedy.py) file contains the code of the greedy algorithm.

The [tsp_ea.py](./tsp_ea.py) file contains the final code of my solution to play with it.

### Run
#### Poetry

If you have [Poetry](https://python-poetry.org/) installed on your pc, you can run the file with the following commands :

```shell
cd CI2024_LAB2 # Go to the directory
```

```shell
poetry install # Install dependencies
```

```shell
poetry shell # Start a shell in the virtual environment
```

```shell
python tsp_ea.py

# Run the program (one of the two)

python tsp_greedy.py
```

#### Others

Otherwise, you can just run the script from scratch.

```shell
cd CI2024_LAB1 # Go to the directory
```

```shell
pip install ... # Install dependencies one by one
```

```shell
python tsp_ea.py

# Run the program (one of the two)

python tsp_greedy.py
```