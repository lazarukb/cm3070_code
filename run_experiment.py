import test_and_run_genetic_algorithm
import datetime
        
def main(parameters):
    print("Hello World!")
    simulation = test_and_run_genetic_algorithm.TestGeneticAlgorithm()
    # run
    simulation.testGeneticAlgorithm(parameters)
    peak_fit = None
    avg_fit = None
    
    return peak_fit, avg_fit
    

if __name__ == "__main__":
    
    # Defaults
    parameters = {
        'experiment_number': '1020',
        'subfolder': '',
        'experiment_comment': "Testing steps to retain and chain rewards",
        'generations': 10,
        'size_new_generations': 10,
        'max_population_size': 11,
        'point_mutation_chance': 0.3,
        'point_mutation_amount': 0.35,
        'point_mutation_chance_max': 0.75,
        'point_mutation_amount_max': 0.5,
        'point_mutation_scalar': 5,
        'game': 'coin_collector_5',
        'steps_to_retain': 10,
        'fitness_bias_scalar': 0.25,
        'failed_step_reward': 0,
        'valid_step_reward': 5,
        'force_random_choice': False,
        'force_pickup': False,
        'chain_rewards': False
    }
    
    
    # For an individual experiment, run these lines
    # Get the timestamp to name the subfolder
    # parameters['subfolder'] = str(datetime.datetime.now().timestamp())
    # main(parameters)
    # quit()
    
    
    # For banks of experiments, comment out above, and run these lines.
    # This is exponential so do be careful with how many choices there are
    # Ranges
    parameter_ranges = {
        'generations': [5],
        'size_new_generations': [10],
        'max_population_size': [15],
        'point_mutation_chance': [0.3],
        'point_mutation_amount': [0.35],
        'point_mutation_chance_max': [0.75],
        'point_mutation_amount_max': [0.5],
        'point_mutation_scalar': [5],
        'game': ['coin_collector_5'],
        'steps_to_retain': [3, 20, 50],
        'fitness_bias_scalar': [0.25],
        'failed_step_reward': [-1],
        'valid_step_reward': [5],
        'force_random_choice': [False],
        'force_pickup': [False],
        'chain_rewards': [False, True]
    }
    
    # Modification loops
    for chain in range(len(parameter_ranges['chain_rewards'])):
        parameters['chain_rewards'] = parameter_ranges['chain_rewards'][chain]
        for f_pickup in range(len(parameter_ranges['force_pickup'])):
            parameters['force_pickup'] = parameter_ranges['force_pickup'][f_pickup]
            for f_random in range(len(parameter_ranges['force_random_choice'])):
                parameters['force_random_choice'] = parameter_ranges['force_random_choice'][f_random]
                for v_step in range(len(parameter_ranges['valid_step_reward'])):
                    parameters['valid_step_reward'] = parameter_ranges['valid_step_reward'][v_step]
                    for f_step in range(len(parameter_ranges['failed_step_reward'])):
                        parameters['failed_step_reward'] = parameter_ranges['failed_step_reward'][f_step]
                        for f_bias in range(len(parameter_ranges['fitness_bias_scalar'])):
                            parameters['fitness_bias_scalar'] = parameter_ranges['fitness_bias_scalar'][f_bias]
                            for s_retain in range(len(parameter_ranges['steps_to_retain'])):
                                parameters['steps_to_retain'] = parameter_ranges['steps_to_retain'][s_retain]
                                for game in range(len(parameter_ranges['game'])):
                                    parameters['game'] = parameter_ranges['game'][game]
                                    for p_m_s in range(len(parameter_ranges['point_mutation_scalar'])):
                                        parameters['point_mutation_scalar'] = parameter_ranges['point_mutation_scalar'][p_m_s]
                                        for p_m_a_m in range(len(parameter_ranges['point_mutation_amount_max'])):
                                            parameters['point_mutation_amount_max'] = parameter_ranges['point_mutation_amount_max'][p_m_a_m]
                                            for p_m_c_m in range(len(parameter_ranges['point_mutation_chance_max'])):
                                                parameters['point_mutation_chance_max'] = parameter_ranges['point_mutation_chance_max'][p_m_c_m]
                                                for p_m_a in range(len(parameter_ranges['point_mutation_amount'])):
                                                    parameters['point_mutation_amount'] = parameter_ranges['point_mutation_amount'][p_m_a]
                                                    for p_m_c in range(len(parameter_ranges['point_mutation_chance'])):
                                                        parameters['point_mutation_chance'] = parameter_ranges['point_mutation_chance'][p_m_c]
                                                        for max_pop in range(len(parameter_ranges['max_population_size'])):
                                                            parameters['max_population_size'] = parameter_ranges['max_population_size'][max_pop]
                                                            for s_new in range(len(parameter_ranges['size_new_generations'])):
                                                                parameters['size_new_generations'] = parameter_ranges['size_new_generations'][s_new]
                                                                for gener in range(len(parameter_ranges['generations'])):
                                                                    parameters['generations'] = parameter_ranges['generations'][gener]
                                                                    
                                                                    # Get the timestamp to name the subfolder
                                                                    parameters['subfolder'] = str(datetime.datetime.now().timestamp())
                                                                    # And run the experiment
                                                                    peak_fit, avg_fit = main(parameters)
                                                                    
                                                                    print(f"peak_fit: {peak_fit}, avg_fit: {avg_fit}")
                                                        
    # Per experiment to change the parameters
    # Number of steps to be retained to the input tensor directly affects the 
    #  inputs to the nn. 
    # main(parameters)
    
    # # Per experiment to change the parameters
    # parameters = {'experiment_number': 1,
    #                 'experiment_comment': "development - no experiment being saved",
    #                 'generations': 10,
    #                 'size_new_generations': 10,
    #                 'max_population_size': 11,
    #                 'point_mutation_chance': 0.3,
    #                 'point_mutation_amount': 0.35,
    #                 'point_mutation_chance_max': 0.75,
    #                 'point_mutation_amount_max': 0.5,
    #                 'point_mutation_scalar': 5,
    #                 'force_random_choice': False,
    #                 'force_pickup': False,
    #                 'game': 'coin_collector_5',
    #                 'steps_to_retain': 5,
    #                 'fitness_bias_scalar': 0.25,
    #                 'failed_step_reward': 0,
    #                 'valid_step_reward': 5,
    #                 'chain_rewards': False}
    # main(parameters)
