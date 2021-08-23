
function brain_initialize(threadID, V, W, T, individual)
    #Fill V,W,T with genome data
    #####################################################
    for i in 1:input_size #range(1:input_size)
        @inbounds V[tx,i] = individuals[blockIdx().x,i+((tx-1)*input_size)]  #tx * input_size
        #@cuprintln("KoordinateV:",tx,";",i,";",blockIdx().x,":",V[tx,i,blockIdx().x]," Genom:",individuals[blockIdx().x,i+((tx-1)*input_size)])
    end
    for i in 1:number_neurons #range(1:number_neurons)
        @inbounds W[tx,i] = individuals[blockIdx().x,v_size+(i+((tx-1)*number_neurons))] #tx * number_neurons
        #@cuprintln("KoordinateW:",tx,";",i,":",W[tx,i])
    end
    for i in 1:output_size #range(1:output_size)
        @inbounds T[i,tx] = individuals[blockIdx().x,v_size+w_size+(tx+((i-1)*number_neurons))] #tx * output_size
        #@cuprintln("KoordinateT:",tx,";",i,":",T[tx,i])
    end
    #####################################################
end

function brain_step(threadID, temp_V, V, W, T, x, input, action)
 #Brain step()
            #############################################
            #V*input matmul:
            V_value = 0.0f0
            for i = 1:input_size 
                @inbounds V_value += V[threadID, i] * input[i] #+ V_value

            end

            @inbounds temp_V[threadID] = tanh(x[threadID] + V_value) #find faster option for this step
            sync_threads()
            #W*temp_V matmul:
            W_value = 0.0f0
            for i = 1:number_neurons 
                @inbounds W_value = W[threadID, i] * temp_V[i] + W_value
            end
            @inbounds x[threadID] += (delta_t * ((-alpha * x[threadID]) + W_value))
            @inbounds x[threadID] = clamp(x[threadID],-clipping_range,clipping_range)
            sync_threads()


            #T*temp_W matmul:
            #=
                T_value = 0.0f0
                for i in 1:output_size
                   @inbounds @atomic action[i] +=  T[i,tx] * x[i]

                end
                if tx <= 2
                @inbounds action[tx] = tanh(action[tx])
                end
                =#
            
            if threadID <= output_size
                T_value = 0.0f0
                for i in 1:number_neurons
                   @inbounds T_value = T_value + T[threadID,i] * x[i]
                end
                @inbounds action[threadID] = tanh(T_value)
            end
            


            #############################################
            #end of Brain step()
end