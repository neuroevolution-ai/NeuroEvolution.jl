####Imports
using CUDA
using Random
#include("brains/continuous_time_rnn.jl")


function kernel_eval_fitness(individuals,results, env_seeds,number_rounds,brain_cfg::CTRNN_Cfg,environment_cfg::Collect_Points_Env_Cfg)

    tx = threadIdx().x
    fitness_total = 0


    V = @cuDynamicSharedMem(Float32,(brain_cfg.number_neurons,environment_cfg.number_inputs))
    W = @cuDynamicSharedMem(Float32,(brain_cfg.number_neurons,brain_cfg.number_neurons),sizeof(V))
    T = @cuDynamicSharedMem(Float32,(environment_cfg.number_outputs,brain_cfg.number_neurons),sizeof(V)+sizeof(W))

    input = @cuDynamicSharedMem(Float32,environment_cfg.number_inputs,sizeof(V)+sizeof(W)+sizeof(T))

    sync_threads()
  
    brain_initialize(tx,blockIdx().x, V,W,T,individuals)

    sync_threads()
    x = @cuDynamicSharedMem(Float32,brain_cfg.number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input))


    temp_V = @cuDynamicSharedMem(Float32,brain_cfg.number_neurons,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(x))
    action = @cuDynamicSharedMem(Float32,environment_cfg.number_outputs,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x))
 
    maze = @cuDynamicSharedMem(Int32,(environment_cfg.maze_columns,environment_cfg.maze_rows,4),sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)) 
    environment_config_array = @cuDynamicSharedMem(Int32,6,sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze))  # Format [agent_x_coordinate,agent_y_coordinate,positive_point_x_coordinate,positive_point_y_coordinate,negative_point_x_coordinate,negative_point_y_coordinate]
    offset = sizeof(V)+sizeof(W)+sizeof(T)+sizeof(input)+sizeof(temp_V)+sizeof(x)+sizeof(action)+sizeof(maze)+sizeof(environment_config_array)

    sync_threads()

    for j in 1:number_rounds
        if threadIdx().x == 1
            @inbounds Random.seed!(Random.default_rng(),env_seeds[blockIdx().x]+j)
        end
        sync_threads()
        #reset brain
        @inbounds x[tx] = 0.0f0
        fitness_current = 0
        
        if tx == 1
            create_maze(maze,environment_cfg, offset)
        end

        #Place agent, positive_point, negative_point randomly in maze
        if tx == 1
            @inbounds environment_config_array[1],environment_config_array[2] = place_agent_randomly_in_maze(environment_cfg)
        end
        if tx == 2
            @inbounds environment_config_array[3],environment_config_array[4] = place_agent_randomly_in_maze(environment_cfg)
        end
        if tx == 3
            @inbounds environment_config_array[5],environment_config_array[6] = place_agent_randomly_in_maze(environment_cfg)
        end
        sync_threads()

        if tx==1
            get_observation(maze,input,environment_config_array,environment_cfg)
        end

        sync_threads()


        #Loop through Timesteps
        #################################################
        for index in 1:environment_cfg.number_time_steps
            #brain_step(tx,temp_V, V, W, T, x, input, action,brain_cfg)
            sync_threads()
            if tx == 1
                #rew = env_step(maze,action,input,environment_config_array,environment_cfg)

                #fitness_current += rew
            end
            sync_threads()
        end

        ####################################################
        #end of Timestep
        if tx == 1
            fitness_total += fitness_current
        end
        sync_threads()

    end
    ######################################################
    #end of Round
    if tx == 1
        @inbounds results[blockIdx().x] = fitness_total / number_rounds
    end
    sync_threads()
    return
end