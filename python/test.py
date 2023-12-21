# main_script.py
import csv
import random
import numpy as np

from src.data_generation import generate_simulation_parameters
from src.functions import roll_dice, can_steal, steal_gift

# Number of players and presents and game timer
num_players = 5
num_presents = 10
game_timer = 1000  # in seconds

# Generate simulation parameters
distance_between_players, friendship_ratings, forgiveness_levels, self_justice_levels = generate_simulation_parameters(num_players)

def run_simulation():
    presents = [[] for _ in range(num_players)]
    retaliation_matrix = np.zeros((num_players, num_players), dtype=int)

    # Open CSV file
    with open("simulation_results.csv", "w", newline='') as file:
        writer = csv.writer(file)
        header = ["Round", "Player", "Dice Roll", "Presents", "Stolen From"] + \
                 [f"Retaliation Player {p + 1}" for p in range(num_players)]
        writer.writerow(header)

        # Distribute presents among players
        all_presents = list(range(num_presents))
        random.shuffle(all_presents)
        for i in range(num_presents):
            presents[i % num_players].append(all_presents[i])

        # Initial distribution of gifts should be saved to the CSV
        for player in range(num_players):
            writer.writerow([0, f"Player {player + 1}", "Initial Distribution", presents[player], "None"] + \
                            [0 for _ in range(num_players)])

        # Game loop
        current_round = 1
        while current_round <= game_timer:
            for player in range(num_players):
                dice_result = roll_dice()
                theft_occurred = False
                stolen_from = None

                # Execute theft logic if dice roll is 6
                if dice_result == 6:
                    theft_candidates = [
                        (can_steal(player, target, presents, distance_between_players,
                                    forgiveness_levels[player], friendship_ratings[player][target],
                                    self_justice_levels[player], retaliation_matrix, current_round), target)
                        for target in range(num_players) if target != player and presents[target]
                    ]
                    theft_candidates.sort(reverse=True)

                    if theft_candidates and theft_candidates[0][0]:
                        target = theft_candidates[0][1]
                        stolen_gift = steal_gift(presents, target)

                        presents[player].append(stolen_gift)
                        presents[target].remove(stolen_gift)
                        theft_occurred = True
                        stolen_from = f"Player {target + 1}"
                        retaliation_matrix[target][player] += 1  # Increment the number of gifts stolen

                # Update retaliation for players who were not stolen from this turn
                for other_player in range(num_players):
                    if theft_occurred and other_player == player:
                        # Skip if the player is the one who stole
                        continue
                    if theft_occurred and other_player != stolen_from:
                        # Only update if the player was the victim of theft
                        continue
                    if retaliation_matrix[other_player][player] > 0:
                        retaliation_matrix[other_player][player] -= 1  # Decrease if there is something to decrease

                # Write round info to the CSV file
                row = [current_round, f"Player {player + 1}", dice_result, presents[player], stolen_from or "None"]
                row.extend(list(retaliation_matrix[player]))
                writer.writerow(row)

            # Increment round number after each player has taken their turn
            current_round += 1

    return presents

# Run the simulation
final_presents = run_simulation()

# Print the final result
for i, gifts in enumerate(final_presents):
    print(f"Player {i + 1}: {gifts}")
