"""Main control for the execution of the genetic algorithm.

Commands the initialisation of the Simulation and Population objects as needed.
Commands the Population to seed itself with initial random networks.
Commands the simulation to evaluate and perform crossover on the Population.
Causes reporting metrics to be created and stored.
"""

import unittest
import population
import simulation
from copy import deepcopy
import numpy as np
import reporting
import breeding

class TestGeneticAlgorithm(unittest.TestCase):
    """Main control for the execution of the genetic algorithm.

    Commands the initialisation of the Simulation and Population objects as needed.
    Commands the Population to seed itself with initial random networks.
    Commands the simulation to evaluate and perform crossover on the Population.
    Causes reporting metrics to be created and stored.
    """
    
    # Initial processing flow of this code was derived, but also fully re-typed
    # and edited, from that presented in lectures in CM3020.
    def test_genetic_algorithm(self, parameters):
        """Main control for the execution of the genetic algorithm.

        Commands the initialisation of the Simulation and Population objects as needed.
        Commands the Population to seed itself with initial random networks.
        Commands the simulation to evaluate and perform crossover on the Population.
        Causes reporting metrics to be created and stored.
    
        Args:
            parameters: DICT of experimental parameters. Most are used here.
             See run_experiment.py for a full description. 

        Returns:
            Float of the average fitness of the generations of this simulation.
        """
        
        # Initialisation
        
        serial_number = 0
        experiment_report = {}
        experiment_results = []
       
        # Extract to shorter variable names from the parameters argument.
        
        generations = parameters['generations']
        size_new_generations = parameters['size_new_generations']
        carryover_count = parameters['carryover_count']
        point_mutation_chance = parameters['point_mutation_chance']
        point_mutation_amount = parameters['point_mutation_amount']
        point_mutation_chance_max = parameters['point_mutation_chance_max']
        point_mutation_amount_max = parameters['point_mutation_amount_max']
        point_mutation_scalar = parameters['point_mutation_scalar']
        force_random_choice = parameters['force_random_choice']
        force_pickup = parameters['force_pickup']
        game = parameters['game']
        fitness_bias_scalar = parameters['fitness_bias_scalar']
        failed_step_reward = parameters['failed_step_reward']
        valid_step_reward = parameters['valid_step_reward']
        chain_rewards = parameters['chain_rewards']
        steps_to_retain = parameters['steps_to_retain']
        
        # Calculated variables derived from the parameters
        # The size of the inputs will be steps * 7
        #  (one for each action, + the choice + the result) + the current
        #  action space length of 5, as described in simulation.py.
        
        inputs_size = (steps_to_retain * 7) + 5
        
        # Early test of directory structure to ensure nothing is overwritten.
        # If this fails then the directory already exists and we don't want to
        #  overwrite previous work, so fail out early before doing a lot of 
        #  calculations and using processing time.
        
        reporting.Reporting.create_folders(parameters)
        
        # Create the initial population of randomised neural networks.
        
        sim_population = population.Population()
        sim_population.create_random_population(
            size_new_generations,
            serial_number,
            inputs_size
            )
        serial_number += size_new_generations
        
        # Validate that the simulation_population is of the proper class, 
        #  and has the specified number of member neural networks
        
        self.assertIsInstance(sim_population, population.Population)
        self.assertEqual(sim_population.get_population_size(), size_new_generations)
        
        # Prepare reporting
        
        experiment_report['parameters'] = parameters
        experiment_report['initial_population'] = reporting.Reporting.census(sim_population)
        experiment_report['generations'] = []
        
        # Create the simulation environment
        
        sim_environment = simulation.Simulation()
        
        # Run the networks through the gym and gather their fitness scores
        
        for iteration in range(generations):
            print(f"Beginning of generation {iteration}. \t", end = "")
            experiment_results = {'after_evaluation': [], 'after_carryover': []}
            self.assertIsInstance(sim_population, population.Population)
            
            # Run the networks through the game, which modifies the 
            #  components of the sim_population object and the network objects
            #  stored in it.
            
            sim_environment.evaluate_population(
                sim_population,
                game,
                force_random_choice,
                force_pickup,
                steps_to_retain,
                failed_step_reward,
                valid_step_reward,
                chain_rewards
                )
            
            # Capture the state of the population with fitnesses after 
            #  they've gone through the evaluation.
            
            experiment_results['after_evaluation'] = reporting.Reporting.census(sim_population)
            
            # Create the fitness map, breed, cross-over, and mutate the 
            #  population to produce a new population.
            sim_population.create_fitness_map()
            new_population = population.Population()
            breeder = breeding.Breeding()
            
            # For each new child neural network that is required, run the
            #  breeder and add the child nn to the new population.
            
            for new_child_nn in range(size_new_generations):
                new_child_nn_obj = breeder.network_cross_and_mutate(
                    sim_population,
                    serial_number,
                    point_mutation_scalar,
                    point_mutation_chance,
                    point_mutation_amount,
                    point_mutation_chance_max,
                    point_mutation_amount_max,
                    fitness_bias_scalar
                    )
                new_population.add_nn(new_child_nn_obj)
                serial_number += 1

            
            # Carryover - retaining most fit network(s) from prev generation.
            # The concept of carryover is not evolution or breeding, so it's 
            #  intentionally not in that class.
            
            if carryover_count != 0:
                # # Get the number of networks to carry over from the old
                # #  generation to the new, capped at the number of networks
                # #  in the old generation.
                
                # size_prev_gen = sim_population.get_population_size()
                # carryover_count = max_population_size - size_new_generations
                # carryover_count = min(carryover_count, size_prev_gen)
                
                # Gather the fitnesses of the previous generation, and use numpy
                #  to create a list sorting the element positions by fitness
                
                all_fitnesses_prev_gen = []
                for i in range(carryover_count):
                    all_fitnesses_prev_gen.append(sim_population.get_nn_fitness(i))
                self.assertEqual(len(all_fitnesses_prev_gen), size_prev_gen)
                all_fitnesses_prev_gen = np.asarray(all_fitnesses_prev_gen)
                most_fit_networks = np.argsort(all_fitnesses_prev_gen)
                most_fit_networks_desc = most_fit_networks[::-1].tolist()

                # Then iterate through the sorted list to copy the specified
                #  count of fit networks to the new generation, unless they have
                #  a minimum fitness, in which case skip them, no point.
                
                for i in range(carryover_count):
                    if sim_population.get_nn_fitness(most_fit_networks_desc[i]) > 1:
                        # Build definition of a network from candidate network
                        
                        import_nn_definition = {}
                        import_nn_definition = sim_population.get_neural_network_def(most_fit_networks_desc[i])
                        self.assertIsInstance(import_nn_definition, dict)
                        import_nn_weights_bias = sim_population.get_weight_bias_definitions(most_fit_networks_desc[i], 1)
                        import_nn_output_weights_bias = sim_population.get_weight_bias_definitions(most_fit_networks_desc[i], 2)
                        import_nn_serial = import_nn_definition['meta']['serial_number']
                        import_nn_parent_1 = import_nn_definition['meta']['parent_1']
                        import_nn_parent_2 = import_nn_definition['meta']['parent_2']
                        self.assertIsInstance(import_nn_weights_bias, list)
                        self.assertIsInstance(import_nn_output_weights_bias, list)
                        
                        # Create object from definition and add to new_population
                        import_nn_obj = sim_population.create_nn(
                            import_nn_serial,
                            import_nn_definition,
                            import_nn_weights_bias,
                            import_nn_output_weights_bias,
                            import_nn_parent_1,
                            import_nn_parent_2
                            )
                        new_population.add_nn(import_nn_obj)
                           
            # Check any carried-over networks against the previous versions to
            #  ensure they are EXACTLY the same from generation to generation.
            
            for i in range(new_population.get_population_size()):
                for new_nn in range(new_population.get_population_size()):
                    new_nn_sn = new_population.get_neural_network_def(new_nn)['meta']['serial_number']
                    for old_nn in range(sim_population.get_population_size()):
                        old_nn_sn = sim_population.get_neural_network_def(old_nn)['meta']['serial_number']
                        if old_nn_sn == new_nn_sn:
                            new_nn_checksum = new_population.get_neural_network_def(new_nn)['meta']['checksum']
                            old_nn_checksum = sim_population.get_neural_network_def(old_nn)['meta']['checksum']
                            self.assertEqual(new_nn_checksum, old_nn_checksum)   
            
            # Replace the old population with the new one
            
            sim_population = deepcopy(new_population)
            new_population = None
            self.assertIsNone(new_population)
            
            # Report the state of the population after they've gone
            #  through the breeding and carryover.
            
            experiment_results['after_carryover'] = reporting.Reporting.census(sim_population)
            experiment_report['generations'].append(experiment_results)
            
            print("Generation is complete.")
        
        # The simulation is complete here
        # Export the reporting
        
        sim_avg_fitness = reporting.Reporting.output_simulation_to_csv(parameters, experiment_report)
        
        return sim_avg_fitness