using Flux

number_inputs = 5

number_time_steps = 200

l = LSTM(3,10)
#lstm_flux = Chain(LSTM(number_inputs, 10), Dense(10, 3))
lstm_flux = LSTM(number_inputs, 10)

par = params(lstm_flux)

A = rand(40,5)

par[1] .= A

# https://stackoverflow.com/questions/59865921/how-can-i-directly-modify-the-weight-values-in-the-julia-library-flux
for p in par
    print(size(p))
    println(typeof(p))
end


for i = 1:number_time_steps

    input = randn(Float32, number_inputs)

    output = lstm_flux(input)
    #@show output
end

print("Finished")