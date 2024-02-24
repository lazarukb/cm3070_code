"""Configures parameters, runs the simulations.

Initial point for configuring the experimental parameters and triggering the
 simulation. Loops through optional lists of parameter values which can run
 a single experiment, or an exponential number of them.
"""

import test_and_run_genetic_algorithm
import datetime
import reporting
        
def main(parameters):
    """Runs the simulation with the parameters argument payload.
    
    Args:
        parameters: dict of the the experimental parameters for this simulation.
         Defined in the initialisation function below.

    Returns:
        avg_fit: float of the average fitness scored across all the generations
         run in this simulation.
    """
    
    # Initialise and run, getting the simulation's average fitness for later.
    
    simulation = test_and_run_genetic_algorithm.TestGeneticAlgorithm()
    avg_fit = simulation.test_genetic_algorithm(parameters)
    
    return avg_fit
    

if __name__ == "__main__":
    # Initialising for some statistics
    
    start = datetime.datetime.now()
    cumulative = {'generations': 0, 'networks' : 0}
    max_fitness = []
    
    # Defaults
    
    parameters = {
        'experiment_number': '1',
        'subfolder': '',
        'experiment_comment': "Testing while editing for formatting",
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
    
    # This is exponential so do be careful with how many choices there are.

    parameter_ranges = {
        'generations': [5],
        'size_new_generations': [5],
        'max_population_size': [6],
        'point_mutation_chance': [0.3],
        'point_mutation_amount': [0.35],
        'point_mutation_chance_max': [0.75],
        'point_mutation_amount_max': [0.5],
        'point_mutation_scalar': [5],
        'game': ['coin_collector_5'],
        'steps_to_retain': [3],
        'fitness_bias_scalar': [0.25],
        'failed_step_reward': [-2],
        'valid_step_reward': [5],
        'force_random_choice': [False],
        'force_pickup': [False],
        'chain_rewards': [False]
    }
    
    # Modification loops
    # Ignoring the 80 character line limit here to ensure this remains both
    #  readable and manually traversable in case of troubleshooting.
    
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
                                                                    
                                                                    avg_fit = main(parameters)
                                                                    
                                                                    # Store max fitness and other stats
                                                                    
                                                                    max_fitness.append([parameters['subfolder'], avg_fit])
                                                                    cumulative['generations'] += parameters['generations']
                                                                    cumulative['networks'] += parameters['size_new_generations']
                                                                    

    
    # Report the folders by highest average fitnesses, and other statistics.
    
    most_fit_folders = sorted(max_fitness, key=lambda x: x[1], reverse=True)
    end = datetime.datetime.now()
    cumulative['elapsed'] = end - start
    cumulative['unique_stamp'] = str(datetime.datetime.now().microsecond)
    print(f"Total duration of the sim: {cumulative['elapsed']}, which is {cumulative['elapsed'] / cumulative['generations']} per generation, or approximately {cumulative['elapsed'] / cumulative['networks']} per network.")
    
    reporting.Reporting.output_runtime_stats_to_csv(most_fit_folders, parameters, cumulative)
    