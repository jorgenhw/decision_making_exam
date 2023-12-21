using ActionModels
using Distributions

function continuous_rescorla_wagner_softmax(agent, input)

    # Read in parameters from the agent
    learning_rate = agent.parameters["learning_rate"]

    # Read in states with an initial value
    old_value = agent.states["value"]

    ##We dont have any settings in this model. If we had, we would read them in as well.
    ##-----This is where the update step starts -------

    ##Get new value state
    new_value = old_value + learning_rate * (input - old_value)


    ##-----This is where the update step ends -------
    ##Create Bernoulli normal distribution our action probability which we calculated in the update step
    action_distribution = Distributions.Normal(new_value, 0.3)

    ##Update the states and save them to agent's history

    agent.states["value"] = new_value
    agent.states["input"] = input

    push!(agent.history["value"], new_value)
    push!(agent.history["input"], input)

    # return the action distribution to sample actions from
    return action_distribution
end

#Define agent

parameters = Dict("learning_rate" => 1, ("initial", "value") => 0)