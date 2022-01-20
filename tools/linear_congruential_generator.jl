
function lgc_random(state)
    next_state = mod(1664525 * state + 1013904223, 2^32)
    return next_state
end

#Returns the next state and a random number with upper bound 
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
function lgc_random(states, index, max)
    next_state = mod(1664525 * states[index] + 1013904223, 2^32)
    states[index] = next_state
    random_number = next_state % (max + 1)
    return random_number
end

function lgc_random(states, index, min, max)
    next_state = mod(1664525 * states[index] + 1013904223, 2^32)
    states[index] = next_state
    random_number = next_state % ((max + 1) - min) + min;
    return random_number
end