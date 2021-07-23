registered_brain_classes = Dict()

#continuous_time_rnn.py implementieren
function get_brain_class(brain_class_name:: String)
    if brain_class_name in keys(registered_brain_classes)
        return registered_brain_classes[brain_class_name]
    else
        return 5#RuntimeError("No valid brain")
    end
end


registered_brain_classes["CTRNN"] = "ContinuousTimeRNN"
