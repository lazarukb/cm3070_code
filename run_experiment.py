"""Configures parameters, runs the simulations.

Initial point for configuring the experimental parameters and triggering the
 simulation. Loops through optional lists of parameter values which can run
 a single experiment, or an exponential number of them.
"""

import test_and_run_genetic_algorithm
import datetime
import reporting
        
def main(hyperparameters):
    """Runs the simulation with the hyperparameters argument payload.
    
    Args:
        hyperparameters: dict of the the experimental hyperparameters for this simulation.
         Defined in the initialisation function below.

    Returns:
        avg_fit: float of the average fitness scored across all the generations
         run in this simulation.
    """
    
    # Initialise and run, getting the simulation's average fitness for later.
    
    simulation = test_and_run_genetic_algorithm.TestGeneticAlgorithm()
    avg_fit = simulation.test_genetic_algorithm(hyperparameters)
    
    return avg_fit
    

if __name__ == "__main__":
    # Initialising for some statistics
    
    start = datetime.datetime.now()
    cumulative = {'generations': 0, 'networks' : 0}
    max_fitness = []
    remaining_experiment_count = 1
    
    # Defaults, collection parameters, fixed values that do not iterate.
    # Don't put commas in the comment!
    
    hyperparameters = {
        'collection_number': '0',
        'experiment': '',
        'collection_comment': "Demonstration video",
    }
    
    # Iterating hyperparameter values.
    # This is exponential so do be careful with how many choices there are.
    
    hyperparameter_ranges = {
        'generations': [10],
        'size_new_generations': [20],
        'carryover_count': [0, 2],
        'point_mutation_chance': [0.3],
        'point_mutation_amount': [0.35],
        'point_mutation_chance_max': [0.75],
        'point_mutation_amount_max': [0.1],
        'point_mutation_scalar': [0.8],
        'game': ['coin_collector_5'],
        'steps_to_retain': [50],
        'fitness_bias_scalar': [0.25],
        'failed_step_reward': [-1],
        'valid_step_reward': [10],
        'force_random_choice': [False],
        'force_pickup': [False],
        'chain_rewards': [False]
    }
    
    for key in hyperparameter_ranges:
        remaining_experiment_count *= len(hyperparameter_ranges[key])
        
    # Confirmation, if case you accidentally queued too many experiments.
    
    confirm = input(f"You've queued up {remaining_experiment_count} " \
        f"experiments, to be saved to collection folder " \
        f"{hyperparameters['collection_number']}. Enter 'n' to cancel, " \
        f"or just press enter to proceed. "
        )
    if confirm.lower() == 'n':
        print(f"Aborting at your request.")
        quit()
        
    # Modification loops to fill in the parameters for a single experiment.
    # Ignoring the 80 character line limit here to ensure this remains both
    #  readable and manually traversable in case of troubleshooting.
    
    for chain in range(len(hyperparameter_ranges['chain_rewards'])):
        hyperparameters['chain_rewards'] = hyperparameter_ranges['chain_rewards'][chain]
        for f_pickup in range(len(hyperparameter_ranges['force_pickup'])):
            hyperparameters['force_pickup'] = hyperparameter_ranges['force_pickup'][f_pickup]
            for f_random in range(len(hyperparameter_ranges['force_random_choice'])):
                hyperparameters['force_random_choice'] = hyperparameter_ranges['force_random_choice'][f_random]
                for v_step in range(len(hyperparameter_ranges['valid_step_reward'])):
                    hyperparameters['valid_step_reward'] = hyperparameter_ranges['valid_step_reward'][v_step]
                    for f_step in range(len(hyperparameter_ranges['failed_step_reward'])):
                        hyperparameters['failed_step_reward'] = hyperparameter_ranges['failed_step_reward'][f_step]
                        for f_bias in range(len(hyperparameter_ranges['fitness_bias_scalar'])):
                            hyperparameters['fitness_bias_scalar'] = hyperparameter_ranges['fitness_bias_scalar'][f_bias]
                            for s_retain in range(len(hyperparameter_ranges['steps_to_retain'])):
                                hyperparameters['steps_to_retain'] = hyperparameter_ranges['steps_to_retain'][s_retain]
                                for game in range(len(hyperparameter_ranges['game'])):
                                    hyperparameters['game'] = hyperparameter_ranges['game'][game]
                                    for p_m_s in range(len(hyperparameter_ranges['point_mutation_scalar'])):
                                        hyperparameters['point_mutation_scalar'] = hyperparameter_ranges['point_mutation_scalar'][p_m_s]
                                        for p_m_a_m in range(len(hyperparameter_ranges['point_mutation_amount_max'])):
                                            hyperparameters['point_mutation_amount_max'] = hyperparameter_ranges['point_mutation_amount_max'][p_m_a_m]
                                            for p_m_c_m in range(len(hyperparameter_ranges['point_mutation_chance_max'])):
                                                hyperparameters['point_mutation_chance_max'] = hyperparameter_ranges['point_mutation_chance_max'][p_m_c_m]
                                                for p_m_a in range(len(hyperparameter_ranges['point_mutation_amount'])):
                                                    hyperparameters['point_mutation_amount'] = hyperparameter_ranges['point_mutation_amount'][p_m_a]
                                                    for p_m_c in range(len(hyperparameter_ranges['point_mutation_chance'])):
                                                        hyperparameters['point_mutation_chance'] = hyperparameter_ranges['point_mutation_chance'][p_m_c]
                                                        for carryover in range(len(hyperparameter_ranges['carryover_count'])):
                                                            hyperparameters['carryover_count'] = hyperparameter_ranges['carryover_count'][carryover]
                                                            for s_new in range(len(hyperparameter_ranges['size_new_generations'])):
                                                                hyperparameters['size_new_generations'] = hyperparameter_ranges['size_new_generations'][s_new]
                                                                for gener in range(len(hyperparameter_ranges['generations'])):
                                                                    hyperparameters['generations'] = hyperparameter_ranges['generations'][gener]
                                                                    
                                                                    # Get the timestamp to name the subfolder
                                                                    
                                                                    exper_start = datetime.datetime.now()
                                                                    hyperparameters['experiment'] = str(exper_start.minute) + "." + str(exper_start.second) + "." + str(exper_start.microsecond)[:4]
                                                                    
                                                                    # And run the experiment
                                                                    
                                                                    avg_fit = main(hyperparameters)
                                                                    
                                                                    # Store max fitness and other stats
                                                                    
                                                                    exper_end = datetime.datetime.now()
                                                                    exper_elapsed = exper_end - exper_start
                                                                    remaining_experiment_count -= 1
                                                                    
                                                                    print(
                                                                        f"Writing experiment {hyperparameters['experiment']} " \
                                                                        f"to the filesystem. That experiment took {exper_elapsed}, " \
                                                                        f"and there are {remaining_experiment_count} experiments left. "\
                                                                        f"Estimating {remaining_experiment_count * exper_elapsed} remaining."
                                                                        )
                                                                    
                                                                    max_fitness.append([hyperparameters['experiment'], avg_fit])
                                                                    cumulative['generations'] += hyperparameters['generations']
                                                                    cumulative['networks'] += hyperparameters['size_new_generations'] * hyperparameters['generations']
                                                                    

    
    # Report the folders by highest average fitnesses, and other statistics.
    
    most_fit_folders = sorted(max_fitness, key=lambda x: x[1], reverse=True)
    end = datetime.datetime.now()
    cumulative['elapsed'] = end - start
    cumulative['unique_stamp'] = str(datetime.datetime.now().microsecond)
    print(
        f"Total duration of the sim: {cumulative['elapsed']}, which is " \
        f"{cumulative['elapsed'] / cumulative['generations']} per generation, " \
        f"or approximately {cumulative['elapsed'] / cumulative['networks']} " \
        f"per network."
        )
    
    reporting.Reporting.output_runtime_stats_to_csv(
        most_fit_folders, 
        hyperparameters, 
        cumulative
        )
    