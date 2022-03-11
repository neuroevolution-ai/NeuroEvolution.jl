#-------------------------------------------------------
#Generic Random Number Generator, usable in GPU execution
#https://de.wikipedia.org/wiki/Kongruenzgenerator
#-------------------------------------------------------

#Without defining a boundary for the random numbers, the state of the RNG equals the current random number
function lgc_random(state::Int)
    next_state = mod(1664525 * state + 1013904223, 2^32)
    return next_state
end

#Returns the next state and a random number with upper bound
#With boundary the random number is calculated based on the RNG state
#The RNG state has to be maintained in order to generate following random numbers
function lgc_random(state::Int, max::Int)
    next_state = mod(1664525 * state + 1013904223, 2^32)
    random_number = next_state % (max + 1)
    return next_state, random_number
end

#Returns the next state and a random number within given range (min, max]
function lgc_random(state::Int, min, max)
    next_state = mod(1664525 * state + 1013904223, 2^32)
    random_number = next_state % ((max + 1) - min) + min;
    return next_state, random_number
end

#For working with multiple RNG-states
#Returns random number in [1, max]
#Here states is an Array, maintaining the RNG states for multiple parallel RNGs
#Following state is stored in the given states array, at the given index
#(Even for using a single RNG it might be recommended to use this method with a 1-dimensional Array and a single entry since the RNG states dont have to be maintained manually)
function lgc_random(states, index, max)
    next_state = mod(1664525 * states[index] + 1013904223, 2^32)
    states[index] = next_state
    random_number = next_state % (max + 1)
    return random_number
end

#Returns random number in [min, max]
#States is Array
function lgc_random(states, index, min, max)
    next_state = mod(1664525 * states[index] + 1013904223, 2^32)
    states[index] = next_state
    random_number = next_state % ((max + 1) - min) + min;
    return random_number
end