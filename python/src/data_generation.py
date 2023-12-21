# generate_matrices.py
import random
import numpy as np

def generate_distance_matrix(num_players):
    distance_matrix = np.zeros((num_players, num_players))
    for i in range(num_players):
        for j in range(num_players):
            if i != j:
                distance_matrix[i, j] = distance_matrix[j, i] = random.randint(50, 200)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix.tolist()

def generate_friendship_matrix(num_players):
    friendship_matrix = np.zeros((num_players, num_players))
    for i in range(num_players):
        for j in range(num_players):
            if i != j:
                friendship_matrix[i, j] = random.randint(1, 7)
    np.fill_diagonal(friendship_matrix, 0)
    return friendship_matrix.tolist()

def generate_simulation_parameters(num_players):
    distance_matrix = generate_distance_matrix(num_players)
    friendship_matrix = generate_friendship_matrix(num_players)
    forgiveness_levels = [random.randint(1, 7) for _ in range(num_players)]
    self_justice_levels = [random.randint(1, 7) for _ in range(num_players)]

    return distance_matrix, friendship_matrix, forgiveness_levels, self_justice_levels
