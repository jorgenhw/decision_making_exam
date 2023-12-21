import random
import numpy as np
import csv

# Number of players and presents and game timer
num_players = 5
num_presents = 10
game_timer = 1000  # in seconds

# Populate matrices and generate other parameters
def generate_simulation_parameters(num_players):
    # Distance matrix
    distance_matrix = np.random.randint(50, 201, size=(num_players, num_players))
    np.fill_diagonal(distance_matrix, 0)  # Set the distance to self as 0

    # Friendship matrix
    friendship_matrix = np.random.randint(1, 8, size=(num_players, num_players))
    np.fill_diagonal(friendship_matrix, 0)  # Update the diagonal to be 0

    forgiveness_levels = np.random.randint(1, 8, size=num_players)
    self_justice_levels = np.random.randint(1, 8, size=num_players)

    return distance_matrix, friendship_matrix, forgiveness_levels, self_justice_levels

# Dice roll
def roll_dice():
    return random.randint(1, 6)

# Determine if the player can steal
def can_steal(player, target, presents, distance_matrix, forgiveness, friendship, justice, retaliation_matrix):
    probability = calculate_steal_probability(player, target, presents, distance_matrix, forgiveness, friendship, justice, retaliation_matrix)
    return random.random() < probability


def calculate_steal_probability(player, target, presents, distance_matrix, forgiveness, friendship, justice, retaliation_matrix):
    distance = distance_matrix[player][target]
    forgiveness_factor = 1 - forgiveness / 7
    retaliation_factor = retaliation_matrix[target][player] / 5

    probability = (-0.1 * friendship / 7 + 0.2 * justice / 7 +
                   0.5 * forgiveness_factor - 0.2 * distance / 200 +
                   0.3 * retaliation_factor)

    if justice > 4 and len(presents[target]) == max(len(presents[p]) for p in range(num_players)):
        probability += 0.1
    return probability

# Choose a gift to steal from the target
def steal_gift(presents, target):
    return random.choice(presents[target])

# The simulation
def run_simulation(num_players, num_presents, game_timer):
    presents = {player: [] for player in range(num_players)}
    retaliation_matrix = np.zeros((num_players, num_players), dtype=int)

    distance_matrix, friendship_matrix, forgiveness_levels, self_justice_levels = generate_simulation_parameters(num_players)

    # Each player gets an equal initial share of presents
    initial_presents = np.arange(num_presents)
    random.shuffle(initial_presents)
    for i, present in enumerate(initial_presents):
        presents[i % num_players].append(present)

    with open("simulation_results.csv", "w", newline='') as file:
        writer = csv.writer(file)
        header = ["Round", "Player", "Dice Roll", "Presents", "Stolen From"] + [f"Retaliation_{p + 1}" for p in range(num_players)]
        writer.writerow(header)
        write_initial_distribution(writer, presents)

        for current_round in range(1, game_timer + 1):
            for player in range(num_players):
                run_round(player, presents, retaliation_matrix, distance_matrix, friendship_matrix, forgiveness_levels, self_justice_levels, writer, current_round)

    return presents

# Helper functions for simulation
def write_initial_distribution(writer, presents):
    for player, gifts in presents.items():
        writer.writerow([0, f"Player {player + 1}", "Initial Distribution", gifts, "None"] + [0] * num_players)

def run_round(player, presents, retaliation_matrix, distance_matrix, friendship_matrix, forgiveness_levels, self_justice_levels, writer, current_round):
    dice_result = roll_dice()
    theft_occurred = False
    stolen_from = None

    if dice_result == 6:
        theft_candidates = get_theft_candidates(player, presents, distance_matrix, forgiveness_levels, friendship_matrix, self_justice_levels, retaliation_matrix)
        if theft_candidates:
            target = theft_candidates[0][1]
            stolen_gift = steal_gift(presents, target)
            presents[player].append(stolen_gift)
            presents[target].remove(stolen_gift)
            theft_occurred = True
            stolen_from = f"Player {target + 1}"
            retaliation_matrix[target][player] += 1

    update_retaliations(player, retaliation_matrix)
    write_round_info(writer, player, dice_result, presents, stolen_from, retaliation_matrix, current_round)

def get_theft_candidates(player, presents, distance_matrix, forgiveness_levels, friendship_matrix, self_justice_levels, retaliation_matrix):
    return sorted(
        [(can_steal(player, target, presents, distance_matrix, forgiveness_levels[player],
                    friendship_matrix[player][target], self_justice_levels[player],
                    retaliation_matrix), target)
        for target in range(num_players) if target != player and presents[target]], reverse=True

    )

def update_retaliations(player, retaliation_matrix):
    for other_player in range(num_players):
        if retaliation_matrix[other_player][player] > 0:
            retaliation_matrix[other_player][player] -= 1

def write_round_info(writer, player, dice_result, presents, stolen_from, retaliation_matrix, current_round):
    row = [current_round, f"Player {player + 1}", dice_result, presents[player], stolen_from or "None"]
    row.extend(retaliation_matrix[player])
    writer.writerow(row)

# Run the simulation
final_presents = run_simulation(num_players, num_presents, game_timer)

# Print the final result
for player, gifts in final_presents.items():
    print(f"Player {player + 1}: {gifts}")
