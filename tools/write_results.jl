using Formatting

function walk_dict(node,callback_node,depth=0)
    for (key,item) in node
        if item isa OrderedDict
            callback_node(key,item,depth,false)
            walk_dict(item,callback_node,depth+1)
        else
            callback_node(key,item,depth,true)
        end
    end
end


function write_results_to_textfile(path, configuration, log, input_size, output_size, individual_size,free_parameter_usage, elapsed_time)

    open(path,"w") do write_file

        function write_dict(key,value,depth,is_leaf)
            pad = ""
            for x in 1:depth
                pad = pad * "\t"
            end
            if is_leaf
                write(write_file,pad*key*": "*string(value))
            else
                write(write_file,pad*key)
            end
            write(write_file,"\n")
        end
    

    walk_dict(configuration,write_dict)

    write(write_file,"\n")
    write(write_file,"Genome Size: $individual_size\n")
    write(write_file,"Free Parameters: "*string(free_parameter_usage)*"\n")
    write(write_file,"Inputs: $input_size\n")
    write(write_file,"Outputs: $output_size\n")
    write(write_file,"\n")
    dash = "------------------------------------------------------------------------------------------"
    write(write_file,dash*"\n")
    
    write(write_file,Formatting.format("{:<8s}{:<14s}{:<14s}{:<14s}{:<14s}{:<14s}\n","gen","min","mean","max","best","elapsed time (s)"))
    write(write_file,dash*"\n")

    #write data for each episode
    for (key,line) in log
        write(write_file,format("{:<8s}{:<14s}{:<14s}{:<14s}{:<14s}{:<14s}\n",line["gen"],line["min"],line["mean"],line["max"],line["best"],line["elapsed_time"]))
    end


    #write elapsed time
    elapsed_time = string(Second(floor(elapsed_time,Second))) * string(elapsed_time % 1000)
    write(write_file,"\nElapsed time for training: $elapsed_time")
    end
    return
end
