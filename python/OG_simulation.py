import random
import numpy as np
import csv

################################################################################
# Number of players and presents and game timer
num_players = 6
num_presents = 10
game_timer = 1000  # in seconds
dice_roll_time = 2 # in seconds
################################################################################



###################################################
#######   Generate simulation parameters   ########
###################################################
# Generate a distance matrix
distance_matrix = np.zeros((num_players, num_players))

# Populating the distance matrix
for i in range(num_players):
    for j in range(num_players):
        if i != j:
            distance_matrix[i, j] = distance_matrix[j, i] = random.randint(50, 200)
np.fill_diagonal(distance_matrix, 0)  # Set the distance to self as 0

# Generate a friendship matrix
friendship_matrix = np.zeros((num_players, num_players))

# Populating the friendship matrix
for i in range(num_players):
    for j in range(num_players):
        if i != j:
            friendship_matrix[i, j] = random.randint(1, 7)
np.fill_diagonal(friendship_matrix, 0)  # Update the diagonal to be 0

# Update the simulation parameters
distance_between_players = distance_matrix.tolist()
friendship_ratings = friendship_matrix.tolist()
forgiveness_levels = [random.randint(1, 7) for _ in range(num_players)]
self_justice_levels = [random.randint(1, 7) for _ in range(num_players)]


###################################################
#######             Dice roll              ########
###################################################
# Helper functions as required, adjusted to apply the correct logic
def roll_dice():
    return random.randint(1, 6)

###################################################
#######         Steal probabilities        ########
###################################################
# Determine if the player can steal based on various factors, including retaliation
def can_steal(player, target, presents, distance_matrix, forgiveness, friendship,
              justice, retaliation_matrix, current_round):
    distance_to_target = distance_matrix[player][target] # Distance between players 
    forgiveness_factor = 1 - forgiveness / 7  # Inverse relation to forgiveness
    retaliation_factor = retaliation_matrix[target][player]  # Number of gifts stolen
    
    # Calculate the probability of stealing based on friendship, fairness, distance, and retaliation
    probability = (
        - 0.1 * friendship / 7 +    # Less likely to steal from friends
        0.2 * justice / 7 +         # More likely to steal if player is "just" (high self_justice_levels)
        0.5 * forgiveness_factor -
        0.2 * distance_to_target / 200 +   # Less likely to steal the further away the target is
        0.3 * retaliation_factor / 5  # Increased likelihood of retaliation based on number of gifts stolen
    )   
    
    if justice > 4 and len(presents[target]) == max(len(presents[p]) for p in range(num_players)):
        probability += 0.1
    
    return random.random() < probability

###################################################
#######         Steal gift func            ########
###################################################
# Choose a gift to steal from the target
def steal_gift(presents, target):
    return random.choice(presents[target])



###################################################
#######           THE SIMULATION           ########
###################################################
def run_simulation():
    
    presents = [[] for _ in range(num_players)]
    retaliation_matrix = np.zeros((num_players, num_players), dtype=int)

    time_remaining = game_timer

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
        while time_remaining > 0:
            for player in range(num_players):
                dice_time = dice_roll_time # the amount each players takes to roll the dice
                time_remaining -= dice_time  # Decrement the game timer
                
                if time_remaining <= 0:
                    # If the time has run out, end the game immediately
                    break

                dice_result = roll_dice()


        ## Game loop
        #current_round = 1
        #while current_round <= game_timer:
        #    for player in range(num_players):
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
