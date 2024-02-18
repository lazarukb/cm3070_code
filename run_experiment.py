import test_and_run_genetic_algorithm
import datetime
        
def main(parameters):
    print("Hello World!")
    simulation = test_and_run_genetic_algorithm.TestGeneticAlgorithm()
    # run
    simulation.testGeneticAlgorithm(parameters)
    

if __name__ == "__main__":
    
    # Defaults
    parameters = {
        'experiment_number': '1000',
        'subfolder': '',
        'experiment_comment': "development - no experiment being saved",
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
    
    # Ranges
    parameter_ranges = {
        'generations': 2,
        'size_new_generations': 5,
        'max_population_size': 6,
        'point_mutation_chance': 0.3,
        'point_mutation_amount': 0.35,
        'point_mutation_chance_max': 0.75,
        'point_mutation_amount_max': 0.5,
        'point_mutation_scalar': 5,
        'game': 'coin_collector_5',
        'steps_to_retain': 10,
        'fitness_bias_scalar': 0.25,
        'failed_step_reward': [-5, -2, -1, 0],
        'valid_step_reward': [1, 3, 5, 7, 10],
        'force_random_choice': [True, False],
        'force_pickup': [True, False],
        'chain_rewards': [True, False]
    }
    
    # Modification loops
    # Booleans first
    for chain in range(len(parameter_ranges['chain_rewards'])):
        parameters['chain_rewards'] = parameter_ranges['chain_rewards'][chain]
        for f_pickup in range(len(parameter_ranges['force_pickup'])):
            parameters['force_pickup'] = parameter_ranges['force_pickup'][f_pickup]
            for f_random in range(len(parameter_ranges['force_random_choice'])):
                parameters['force_random_choice'] = parameter_ranges['force_random_choice'][f_random]
                
                # Get the timestamp to name the subfolder
                parameters['subfolder'] = str(datetime.datetime.now().timestamp())
                # print(f"{parameters['subfolder']}")
                # quit()
                main(parameters)
    
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
