# loading the packages #
using ActionModels
using Distributions
using Plots
using DataFrames

###############################################
######## creating the action model ############
###############################################

# Defining the softmax function 
softmax(x) = exp.(x) ./    
sum(exp.(x))

# initializing our action model/decision model

function simple_action_model(agent, input)
    """
    This function takes an agent (an agent structure) and the input (as array/vector of vectors)
    """

    # To take out parameters from the agent
    orientation = agent.parameters["self_vector"]
    self_number = agent.parameters["self_number"]
    hatefullness = agent.parameters["hatefullness"]
    distance_penalty = agent.parameters["distance_penalty"]
    
    # Get the distances for the agent in question
    

    # take out the gift amounts (the first element of our input) and multiply with the agents "self" to 
    # make sure that the agent does not steal packages from themselves
    gift_amounts = input[1] .* orientation
    #gift_unique = input[3] .* orientation
    distance_matrix = input[3]
    
    # Get the distances for the agent in question
    distances = distance_matrix[self_number, :]

    # Apply the distance penalty to reduce the attractiveness of distant gifts
    distance_effects = (-distances .* distance_penalty) .* orientation

    # Apply the distance effects to the gift amounts
    gift_amounts = gift_amounts .+ distance_effects


    # convert the vector to "Real" data type instead of Int64 something...
    # - it caused some problems earlier, so just to make sure
    gift_amounts = convert(Vector{Real}, gift_amounts) 
   
    # calculate "hate vector# from the second part of input, the matrix, and take out the agents own "hate row".
    # the hate row is multiplied with the hatefullness to "scale up" the contribution of hate
    hate_vector = input[2][self_number,:] .* hatefullness

    # the gift amounts are added with the contribution from the hate vector
    gift_amounts = hate_vector .+ gift_amounts
    gift_amounts = max.(gift_amounts, 0)
    # the gifts are put into a softmax (this vector has to be positive, it can cause some problems when recovering -
    # - parameters  if the vector is negative in some entries)

    max_value, index = findmax(gift_amounts)

    #calculate the action distribution with the categorical.
    action_distribution = Dirac(index)

    # print type of action_probalities_2 
    
    #returning the categorical distribution as the action distribution
    return action_distribution
end


########################################
######## defining the agent ############
########################################

# Define the distance penalty vector (for simplicity we're assuming it's constant for now)
distance_penalty = Real(1.3)  # you can adjust this value to suit your game dynamics

id_1 = ones(Int8, 4)        # id vector for agent 1

id_1[1] = 0                 # placing a 0 on the first entry
convert(Vector{Real}, id_1) # transform this vector to "Real" datatype (in order to add and subtract values from it)

# initialize agent. the input to init_agent is the action model, a dictionary of parameters, and a list of states (if we want to save anything...assume we don't)
agent_1 = init_agent(simple_action_model, 
                    parameters = Dict("self_vector" => id_1, #set the different parameters (can be changed later)
                                    "self_number" => 1,
                                    "hatefullness" => Real(1),
                                    "distance_penalty" => distance_penalty
                                    ))

distance_matrix = [
    0 0.1 0.2 0.3;
    0.1 0 0.1 0.2;
    0.2 0.1 0 0.1;
    0.3 0.2 0.1 0
]

########################################
######## creating the data #############
########################################

""" 
structure of data is    [
                        [[list of packages], [hatematrix]]
                        [[list of packages], [hatematrix]]
                        [[list of packages], [hatematrix]]
                        [[list of packages], [hatematrix]]
                        ]  

like this: 
[ [ [package1, package2, package3, package4], [row1 ; row2 ; row3 ; row 4] ] , ...] 

remember: the id of the agent (eg. agent 1) means that the first entry of both the
hatematrix and the package vector is always 0...

"""

# test input of size 15 (just handmade data)
test_input = [[[1,1,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix], # only the first row of the matrix is filled since we are working with agent_1
              [[1,1,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,1,2,1],[0 0 0 1;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,1,3,1],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,2,1,4],[0 0 0 1;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,3,2,2],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,2,5,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,3,1,2],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,3,0,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,3,1,2],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,1,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,1,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,1,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,2,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              [[1,1,1,3],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0], distance_matrix],
              ]

# set the hatefullness parameter here to the value we try to recover
set_parameters!(agent_1, Dict("hatefullness" => Real(2)))


#reset the agent if it has recieved input before (cleanse the plate)
reset!(agent_1)

# give the input and recieve actions
actions = give_inputs!(agent_1,test_input)

actions



################
# parameter recovery
################

# set the prior of hatefullness
priors = Dict(
    "hatefullness" => Normal(0,2),
    "distance_penalty" => Uniform(0, 3)   # Adjust the mean and std. dev. as needed)
    )

# run the model 
fitted_model_agent_1 = fit_model(agent_1, priors, test_input, actions, n_chains = 2)



######### IDEA OF A TO DO ##########
""" 

set different parameters (maybe range from 0 to 5) and see how well the model recovers... 
maybe run 5 recoveries pr. parameter test and log these to make a "mean" or an idea of how often it 
hits the right value in different parameter spaces


something like:

hatefullness      parameter recovered           "hit rate" mean
0.01              0.2 , 0.4, 0.1, 0.01           0.2
0.1               0.3 , 0.4, 0.1, 0.8            0.7
bla bla bla       bla bla bla                    bla bla


To get an idea of how well this simple model performs...

"""