using Random
using Distances
using CSV

####################################################################################
# Number of players and presents and game timer
num_players = 6
num_presents = 10
game_timer = 1000 # in seconds
dice_roll_time = 2 # in seconds
####################################################################################

# Generate a distance matrix
distance_matrix = zeros(Int, num_players, num_players)

# Populating the distance matrix
for i in 1:num_players, j in 1:num_players
    if i != j
        distance_matrix[i, j] = rand(50:200)
    end
end

distance_matrix

# Generate a friendship matrix
friendship_matrix = zeros(Int, num_players, num_players)

# Populating the friendship matrix
for i in 1:num_players, j in 1:num_players
    if i != j
        friendship_matrix[i, j] = rand(1:7)
    end
end

forgiveness_levels = [rand(1:7) for _ in 1:num_players]
self_justice_levels = [rand(1:7) for _ in 1:num_players]

# Helper function to roll a dice
roll_dice() = rand(1:6)



# Stealing logic
function can_steal(player, target, presents, distance_matrix, forgiveness, friendship, justice,
    retaliation_matrix, current_round)
    distance_to_target = distance_matrix[player, target]
    forgiveness_factor = 1 - forgiveness / 7
    retaliation_factor = retaliation_matrix[target, player]
    
    probability = (
        - 0.1 * friendship / 7 +
        0.2 * justice / 7 +
        0.5 * forgiveness_factor -
        0.2 * distance_to_target / 200 +
        0.3 * retaliation_factor / 5
    )
    
    if justice > 4 && length(presents[target]) == maximum(length.(presents))
        probability += 0.1
    end
    
    return rand() < probability
end

# Choose a gift to steal from the target
steal_gift(presents, target) = rand(presents[target])

# Running the simulation
function run_simulation()
    presents = [[] for _ in 1:num_players]
    retaliation_matrix = zeros(Int, num_players, num_players)
    time_remaining = game_timer
    
    CSV.open("simulation_results.csv", "w") do file
        header = ["Round", "Player", "Dice Roll", "Presents", "Stolen From"]
        append!(header, ["Retaliation Player $p" for p in 1:num_players])
        writeheader = CSV.Header(header)
        write(file, writeheader)

        # Distribute presents among players
        all_presents = shuffle(collect(1:num_presents))
        for i in 1:num_presents
            push!(presents[(i-1) % num_players + 1], all_presents[i])
        end

        # Game loop
        current_round = 1
        while time_remaining > 0
            for player in 1:num_players
                dice_time = dice_roll_time
                time_remaining -= dice_time
                
                if time_remaining <= 0
                    break
                end

                dice_result = roll_dice()
                theft_occurred = false
                stolen_from = "None"

                # Execute theft logic if dice roll is 6
                if dice_result == 6
                    theft_candidates = [
                        (can_steal(player, target, presents, distance_matrix, forgiveness_levels[player],
                        friendship_matrix[player, target], self_justice_levels[player], retaliation_matrix,
                        current_round), target) for target in 1:num_players if target != player && !isempty(presents[target])
                    ]
                    sort!(theft_candidates, by=x->x[1], rev=true)

                    if !isempty(theft_candidates) && theft_candidates[1][1]
                        target = theft_candidates[1][2]
                        stolen_gift = steal_gift(presents, target)
                        push!(presents[player], stolen_gift)
                        filter!(gift -> gift != stolen_gift, presents[target])
                        theft_occurred = true
                        stolen_from = "Player $(target)"
                        retaliation_matrix[target, player] += 1
                    end
                end

                # Update retaliation for players who were not stolen from this turn
                for other_player in 1:num_players
                    if theft_occurred && other_player == player
                        continue
                    elseif theft_occurred && other_player != stolen_from
                        continue
                    elseif retaliation_matrix[other_player, player] > 0
                        retaliation_matrix[other_player, player] -= 1
                    end
