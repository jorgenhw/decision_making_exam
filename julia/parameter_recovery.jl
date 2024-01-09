# loading the packages #
using ActionModels
using Distributions
using Plots
using DataFrames
using NNlib
using StatsPlots
###############################################
######## creating the action model ############
###############################################

# initializing our action model/decision model

function simple_action_model_dirac(agent, input)
    """
    This function takes an agent (an agent structure) and the input (as array/vector of vectors)
    """

    # To take out parameters from the agent.
    orientation = agent.parameters["self_vector"]
    self_number = agent.parameters["self_number"]
    hatefullness = agent.parameters["retaliation"]
    proximity = agent.parameters["proximity"]
    distance = [0, 0.8, 0.1, 0.5, 0.3, 0.8, 0.1, 0.4] # polynomial fit...

    # take out the gift amounts (the first element of our input) and multiply with the agents "self" to 
    # make sure that the agent does not steal packages from themselves.
    gift_amounts = input[1] .* orientation 
    gift_amounts = gift_amounts .- (proximity .* distance)

    # convert the vector to "Real" data type instead of Int64 something.
    # - it caused some problems earlier, so just to make sure.
    gift_amounts = convert(Vector{Real}, gift_amounts)
   
    # calculate "hate vector# from the second part of input, the matrix, and take out the agents own "hate row".
    # the hate row is multiplied with the hatefullness to "scale up" the contribution of hate.
    hate_vector = input[2] .* hatefullness

    # the gift amounts are added with the contribution from the hate vector.
    gift_amounts = hate_vector .+ gift_amounts

    # the gifts are put into a softmax (this vector has to be positive, it can cause some problems when recovering -
    # - parameters  if the vector is negative in some entries).
    max_value, index = findmax(gift_amounts)

    #calculate the action distribution with the categorical.
    action_distribution = Dirac(index)

    #returning the categorical distribution as the action distribution.
    return action_distribution
end

# softmax log normal prior... 


# kill the rich parameter: ikke lineær skalering,(måske softmax)
# add noiiiiise 


function simple_action_model_softmax(agent, input)
    """
    This function takes an agent (an agent structure) and the input (as array/vector of vectors)
    """

    # To take out parameters from the agent.
    orientation = agent.parameters["self_vector"]
    self_number = agent.parameters["self_number"]
    hatefullness = agent.parameters["retaliation"]
   # proximity = agent.parameters["proximity"]
   # distance = [0, 0.8, 0.1, 0.5, 0.3, 0.8, 0.1, 0.4] # polynomial fit...

    # take out the gift amounts (the first element of our input) and multiply with the agents "self" to 
    # make sure that the agent does not steal packages from themselves.
    gift_amounts = input[1] .* orientation 
   # gift_amounts = gift_amounts .- (proximity .* distance)

    # convert the vector to "Real" data type instead of Int64 something.
    # - it caused some problems earlier, so just to make sure.
    gift_amounts = convert(Vector{Real}, gift_amounts)
   
    # calculate "hate vector# from the second part of input, the matrix, and take out the agents own "hate row".
    # the hate row is multiplied with the hatefullness to "scale up" the contribution of hate.
    hate_vector = input[2] .* hatefullness

    # the gift amounts are added with the contribution from the hate vector.
    gift_amounts = hate_vector .+ gift_amounts

    softmax_gift = NNlib.softmax(gift_amounts)

    #calculate the action distribution with the categorical.
    action_distribution = Categorical(softmax_gift)

    #returning the categorical distribution as the action distribution.
    return action_distribution
end

########################################
######## creating the agent ############
########################################



id_1 = ones(Int8, 8)        # id vector for agent 1
id_1[1] = 0                 # placing a 0 on the first entry 
convert(Vector{Real}, id_1) # transform this vector to "Real" datatype (in order to add and subtract values from it)

# initialize agent. the input to init_agent is the action model, a dictionary of parameters,
# and a list of states (if we want to save anything...assume we dont)

agent_1 = init_agent(simple_action_model_softmax, 
                    parameters = Dict("self_vector" => id_1, #set the different parameters (can be changed later)
                                    "self_number" => 1, #add to settings..
                                    "retaliation" => Real(1),
                                    "proximity" => Real(2) #add eat the rich 
                                    ))

########################################
######## creating the data #############
########################################

""" 
structure of data is [[list of packages, hatematrix] , [list of packages, hatematrix] ... ]

like this: 
[ [ [package1, package2, package3, package4], [row1 ; row2 ; row3 ; row 4] ] , ...] 

remember: the id of the agent (eg. agent 1) means that the first entry of both the
hatematrix and the package vector is always 0...

Round 1 only:
p1 tjek
p2
p3
p4
p5
p6
p7
p8
"""

### REAL DATA ####
actions = convert(Vector{Any},[5, 1, 1, 6, 4, 5, 4, 4, 7, 5, 3, 1, 2, 6, 3, 1, 5])

test_input_r1 = [[[2, 0, 1, 0, 3, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0]],
[[2, 1, 2, 0, 2, 1, 2, 2], [0, 0, 1, 0, 0, 0, 0, 0]],
[[3, 3, 2, 1, 0, 2, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0]],
[[1, 2, 2, 2, 0, 2, 2, 1], [0, 0, 0, 0, 1, 0, 0, 0]],
[[0, 3, 2, 2, 1, 0, 2, 2], [0, 0, 0, 0, 1, 0, 0, 0]],
[[1, 1, 2, 1, 2, 1, 2, 2], [0, 0, 1, 0, 0, 0, 0, 0]],
[[1, 1, 1, 2, 1, 1, 2, 3], [0, 0, 1, 0, 0, 0, 0, 0]],
[[1, 2, 1, 1, 1, 1, 2, 3], [0, 1, 0, 0, 0, 0, 0, 0]],
[[1, 3, 1, 0, 1, 1, 3, 2], [0, 0, 0, 0, 0, 0, 1, 0]],
[[1, 2, 2, 0, 3, 2, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0]],
[[1, 1, 2, 1, 2, 2, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0]],
[[2, 3, 1, 1, 1, 2, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0]],
[[2, 3, 1, 1, 2, 1, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0]],
[[2, 1, 2, 0, 2, 3, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0]],
[[2, 1, 2, 1, 1, 2, 1, 2], [0, 0, 1, 0, 0, 0, 0, 0]],
[[2, 2, 1, 1, 1, 2, 1, 2], [0, 1, 0, 0, 0, 0, 0, 0]],
[[2, 3, 1, 1, 1, 1, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0]]]
#reset the agent if it has recieved input before (cleanse the plate)
reset!(agent_1)

# set the prior of hatefullness
priors = Dict("retaliation" => Uniform(-5,5))

# run the model 
fitted_model_agent_1 = fit_model(agent_1, priors, test_input_r1, actions, n_chains = 2)

plot_parameter_distribution(fitted_model_agent_1, priors)

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


test_input_30 = [[[1,1,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]], # only the first row of the matrix is filled since we are working with agent_1
            [[1,1,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,2,1],[0 0 0 1;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,3,1],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,2,1,4],[0 0 0 1;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,2,2],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,2,5,4],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,1,2],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,0,1],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,1,2],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,1,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,2,1],[0 0 0 1;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,1,4],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,2,1,1],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,1,3],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,1,1],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,1,3],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[3,1,3,4],[0 0 0 1;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,3,1],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,5,1],[0 0 0 1;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,3,2],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,2,5,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[12,3,4,4],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,0,5],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,1,2],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,2,1,3],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,4,1,3],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,5,1],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,3,1,1],[0 1 0 0;0 0 0 0;0 0 0 0;0 0 0 0]],
            [[1,1,4,3],[0 0 1 0;0 0 0 0;0 0 0 0;0 0 0 0]],
             ] 
    
"""        

posteriors = []

for i in 1:5 
        # set the hatefullness parameter here to the value we try to recover
    set_parameters!(agent_1, Dict("hatefullness" => Real(3)))

    #reset the agent if it has recieved input before (cleanse the plate)
    reset!(agent_1)

    # give the input and recieve actions
    actions = give_inputs!(agent_1,test_input_p1_r1)

    # set the prior of hatefullness
    priors = Dict("hatefullness" => Normal(1,3))

    # run the model 
    fitted_model_agent_1 = fit_model(agent_1, priors, test_input, actions, n_chains = 2)

    push!(posteriors,get_posteriors(fitted_model_agent_1))

end
posteriors

valuess = []
for i in 1:5
    push!(valuess, posteriors[i]["hatefullness"])
end

mean(valuess)

Categorical(softmax([1, 5, 0, 3, 4, 3]))
rand(Categorical(softmax([1, 5, 0, 3, 4, 3])),10)


 ######## ######## ######## ######## ######## ########
   ########## Implementing Andreas' requests #######
 ######## ######## ######## ######## ######## ########

"""
1. Har I prøvet at køre med samme prior for alle recoveries? (uniform priors)
2. 100 hate-værdier for 15 og 30 trials-modellerne og plotte et scatterplot?
3. Har I mulighed for at rapportere 95% credible intervals frem for sd… 

Below: running test on true parameter 0.5

True parameter range test
0.5
1
1.5
2
2.5
3

 """

posteriors = []
summary_stats_all =[]
quantieles_all = []

posteriors_30 = []
summary_stats_all_30 =[]
quantieles_all_30 = []

for test_param in [0.5, 1, 1.5, 2, 2.5, 3]
    print(test_param)
    for i in 1:100
            # set the hatefullness parameter here to the value we try to recover
        set_parameters!(agent_1, Dict("hatefullness" => test_param))

        #reset the agent if it has recieved input before (cleanse the plate)
        reset!(agent_1)

        # give the input and recieve actions
        actions = give_inputs!(agent_1,test_input_p1_r1)

        # set the prior of hatefullness
        priors = Dict("hatefullness" => Uniform(0,7))

        # run the model 
        fitted_model_agent_1 = fit_model(agent_1, priors, test_input_p1_r1, actions, n_chains = 2)
        
        summary_stats, quantieles = describe(fitted_model_agent_1)
        push!(summary_stats_all_30,summary_stats)
        push!(quantieles_all_30,quantieles)
        push!(posteriors_30,get_posteriors(fitted_model_agent_1))

    end
end

posteriors_30
summary_stats_all_30
quantieles_all_30


values_all_30=[]
for i in 1:600
    push!(values_all_30, posteriors_30[i]["hatefullness"])
end

quantiles_all = []
for i in 1:600
    push!(quantiles_all,quantieles_all_30[1][1,:])
end

summary_stats_15_column = summary_stats_all_15
#posterior_15_column = values_all_15
#posterior_30_column = values_all_30



true_labels = [0.5, 1, 1.5, 2, 2.5, 3]
true_label = []
for i in 1:6
    current_label = true_labels[i]
    for j in 1:100
        push!(true_label,current_label)
    end
end
true_label


df2 = DataFrame(posteriors = values_all_30, true_labels = true_label)


df2[!,:posteriors] = convert.(Real,df2[!,:posteriors])

df2[!,:true_labels] = convert.(Real,df2[!,:true_labels])

using CSV, DataFrames


CSV.write("crap_30.csv", df2) 

