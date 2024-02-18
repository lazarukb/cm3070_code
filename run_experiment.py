import test_and_run_genetic_algorithm     
        
def main(parameters):
    print("Hello World!")
    simulation = test_and_run_genetic_algorithm.TestGeneticAlgorithm()
    # run
    simulation.testGeneticAlgorithm(parameters)
    

if __name__ == "__main__":
    
    # Per experiment to change the parameters
    # Number of steps to be saved to the input tensor directly affects the 
    #  inputs to the nn. The size of the inputs will be steps * 7
    #  (one for each action, + the choice + the result) + the current
    #  action space length of 5, as described in simulation.py.
    parameters = {'experiment_number': 1,
                    'experiment_comment': "development - no experiment being saved",
                    'generations': 10,
                    'size_new_generations': 10,
                    'max_population_size': 11,
                    'point_mutation_chance': 0.3,
                    'point_mutation_amount': 0.35,
                    'point_mutation_chance_max': 0.75,
                    'point_mutation_amount_max': 0.5,
                    'point_mutation_scalar': 5,
                    'force_random_choice': False,
                    'force_pickup': False,
                    'game': 'coin_collector_5',
                    'steps_to_retain': 10,
                    'fitness_bias_scalar': 0.25,
                    'failed_step_reward': 0,
                    'valid_step_reward': 5,
                    'chain_rewards': False}
    main(parameters)
    
    # Per experiment to change the parameters
    parameters = {'experiment_number': 1,
                    'experiment_comment': "development - no experiment being saved",
                    'generations': 10,
                    'size_new_generations': 10,
                    'max_population_size': 11,
                    'point_mutation_chance': 0.3,
                    'point_mutation_amount': 0.35,
                    'point_mutation_chance_max': 0.75,
                    'point_mutation_amount_max': 0.5,
                    'point_mutation_scalar': 5,
                    'force_random_choice': False,
                    'force_pickup': False,
                    'game': 'coin_collector_5',
                    'steps_to_retain': 5,
                    'fitness_bias_scalar': 0.25,
                    'failed_step_reward': 0,
                    'valid_step_reward': 5,
                    'chain_rewards': False}
    main(parameters)
