# simulation_functions.py
import random

def roll_dice():
    return random.randint(1, 6)

def can_steal(player, target, presents, distance_matrix, forgiveness, friendship,
              justice, retaliation_matrix, current_round):
    distance_to_target = distance_matrix[player][target]
    forgiveness_factor = 1 - forgiveness / 7
    retaliation_factor = retaliation_matrix[target][player]
    
    probability = (
        - 0.1 * friendship / 7 +
        0.2 * justice / 7 +
        0.5 * forgiveness_factor -
        0.2 * distance_to_target / 200 +
        0.3 * retaliation_factor / 5
    )   
    
    if justice > 4 and len(presents[target]) == max(len(presents[p]) for p in range(len(presents))):
        probability += 0.1
    
    return random.random() < probability

def steal_gift(presents, target):
    return random.choice(presents[target])
