
#=
data eneded for episode runner:

self.env_class = env_class
    self.env_configuration = env_configuration
    env = self.env_class(env_seed=0, configuration=self.env_configuration)
    self.input_size = env.get_number_inputs()
    self.output_size = env.get_number_outputs()

    self.brain_class = brain_class
    self.brain_configuration = brain_configuration

    self.brain_state = brain_class.generate_brain_state(input_size=self.input_size,
                                                        output_size=self.output_size,
                                                        configuration=self.brain_configuration)

=#
struct EpisodeRunner

    env_class ::
    env_configuration :: Dict
    environment ::


end

function eval_fitness()
    body
end

function kernel_eval_fitness()

end
