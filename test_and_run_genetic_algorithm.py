import unittest
# import genome
import population
import simulation_utest
from copy import deepcopy
import random
import numpy as np
import reporting
from operator import itemgetter
import breeding

# So this is the main test-and-run ... all the unittesting stays in here.  Remove from elsewhere.


# To remove unittest, if eventually decided ...
# Make sure the files are open (maybe, but just to be sure)
# Remove the unittest.TestCase argument and unittest.main() (if exists) calls from the module
# Rename the filename of the class module file

class TestGeneticAlgorithm(unittest.TestCase):
    
    # def __init__(self):
    
    # Code derived, but also fully re-typed and edited, from that presented in lectures in CM3020
    def testGeneticAlgorithm(self):
        # Define the population, simulation, and evolution parameters
        experiment_number = 1
        # This is also written as a folder in the experiment folder for quick 
        #   reference, so keep it short.
        experiment_comment = "development - no experiment being saved"
        serial_number = 0
        populations = []             ###### for possible future use in saving multiple populations for comparisons
        generations = 5
        size_new_generations = 30
        max_population_size = 32
        point_mutation_chance = 0.1
        point_mutation_amount = 0.05
        point_mutation_chance_max = 0.75
        point_mutation_amount_max = 0.5
        point_mutation_scalar = 5
        shrink_mutation_chance = 0.25
        growth_mutation_chance = 0.25
        force_random_choice = False
        force_pickup = False
        experiment_report = {}
        experiment_results = []
        game = 'coin_collector_5'
        
        # Number of steps to be saved to the input tensor directly affects the inputs to the nn.
        # So the inputs needs to be removed from the base genome as it's variable now.
        # And the size of the inputs will be steps * 7 (one for each action, + the choice + the result) + the current action space of 5.
        steps_to_retain = 2
        inputs_size = (steps_to_retain * 7) + 5
        
        # possibly mod this to read experiments from a CSV and then iterate through them.
        # LAST
               
        # helpers_utest.TestHelpers.testVerifyParameters(self)
        
        # Early test of the directory structure to ensure nothing is overwritten.
        reporting.Reporting.create_folders(experiment_number, experiment_comment)
        
        # Create the initial population of randomised neural network definitions
        sim_population = population.Population()
        sim_population.create_random_population(size_new_generations, serial_number, inputs_size)
        serial_number += size_new_generations
        
        # Validate that the simulation_population is of the proper class, and has the specified number of member neural networks
        self.assertIsInstance(sim_population, population.Population)
        # self.assertEqual(population.Population.get_population_size(sim_population), size_new_generations)
        self.assertEqual(sim_population.get_population_size(), size_new_generations)
        
        # Document the initial state
        experiment_report['parameters'] = {
            'experiment_comment': experiment_comment,
            'generations': generations, 
            'size_new_generations': size_new_generations, 
            'max_population_size': max_population_size,
            'point_mutation_chance': point_mutation_chance,
            'point_mutation_amount': point_mutation_amount,
            'force_random_choice': force_random_choice,
            'force_pickup': force_pickup,
            'game': game
            }
        
        experiment_report['initial_population'] = reporting.Reporting.census(sim_population)
        experiment_report['generations'] = []
        
        # Create the simulation environment
        sim_environment = simulation_utest.Simulation()
        
        # Run the networks through the gym and gather their fitness scores
        for iteration in range(generations):
            print(f"Beginning of evaluation for generation {iteration}, ", end = "")
            # experiment_report['generations'].append([])
            experiment_results = {'after_evaluation': [], 'after_carryover': []}
                       
            # Validate that the simulation_population is of the proper class, and has the specified number of member neural networks
            self.assertIsInstance(sim_population, population.Population)
            
            # run the networks through the game, which modifies the components of the simulation_population object
            ############################## WAIT, WHAT IS THIS MAX STEPS?
            sim_environment.evaluate_population(sim_population, game, force_random_choice, force_pickup, steps_to_retain)
            
            # Capture the state of the population with fitnesses after they've gone through the evaluation
            experiment_results['after_evaluation'] = reporting.Reporting.census(sim_population)
            
            # get the fitnesses for each network
            # networks_count = population.TestPopulation.get_population_size(simulation_population)
            # all_fitnesses = []
            # for i in range(networks_count):
            #     all_fitnesses.append(population.TestPopulation.get_nn_fitness(simulation_population, i))
            
            # generate and display statistics for the generation
            
            # print(f"create and test the fitness map ", end = "")
            # create the fitness map for this population
            sim_population.create_fitness_map()
            
            # Test the fitness map for a minimum of two networks with > 0 fitness
            # Take the value of the last element in the map, which is guaranteed to be the largest because of how the map is created.
            # Now divide that by the length of the list.
            # Then take the average of all the values in the list.
            # If these are equal then there are 0 or 1 networks with a fitness.
            # No breeding can happen. Throw out the generation and randomise a new one.
            # If these are not equal, we have at least two individuals, so breeding can commence.
            # print(f"Testing the fitness map")
            # fitness_map = sim_population.get_fitness_map()
            # marker1_max_fitness = max(fitness_map) / len(fitness_map)
            # marker2_avg_fitness = sum(fitness_map) / len(fitness_map)
            
            # print(fitness_map)
            # print(f"marker1_max_fitness: {marker1_max_fitness}")
            # print(f"marker2_avg_fitness: {marker2_avg_fitness}")
            
                    
            # capture the DNA of the elite network from this generation
            
            # update, if necessary, the elite network from this simulation
            
            
            # Validations of the current population before we get into breeding
            # print("\nValidations")
            # print("1. Serial numbers")
            
            # for i in range(init_population_size):
            #     print(population.Population.get_neural_network_def(sim_population, i)["meta"])
            #     print(population.Population.get_nn_fitness(sim_population, i))
            #     print(population.Population.get_neural_network_def(sim_population, i)["input"])
            #     print(population.Population.get_weight_bias_definitions(sim_population, i, 1)[0])
                        
            # Create new networks through breeding two randomly selected parent networks, and incorporating mutation
            # if marker1_max_fitness != marker2_avg_fitness:
            
            # Start with picking the two parents and sending those to a helper, returning the new child.            
            
            # Breed, cross-over, and mutate the population to produce a new population
            print(f"Proceed with breeding ", end = "")
            # Create the population instance target for the children
            new_population = population.Population()
            # Create an instance to handle the breeding and evolution work
            breeder = breeding.Breeding()
            for generation in range(size_new_generations):
                new_population = breeder.cross_and_mutate(sim_population, new_population, serial_number, point_mutation_scalar, point_mutation_chance, point_mutation_amount, point_mutation_chance_max, point_mutation_amount_max)
                serial_number += 1

            
            # If you're going to retain the most fit individual(s) from the previous generation, here is where that should happen
            # select the most fit individuals from the existing generation and move them into the new generation
            if size_new_generations != max_population_size:
                # Get the number of networks to carry over from the old generation to the new
                # But cap at the number of networks in the old generation.
                
                print(f"\nProduce carryover from the old generation to the new. At most ", end = "")
                # size_prev_gen = population.Population.get_population_size(sim_population)
                size_prev_gen = sim_population.get_population_size()
                carryover_count = max_population_size - size_new_generations
                carryover_count = min(carryover_count, size_prev_gen)
                print(f"{carryover_count} networks from the old generation could import to the new generation.")
                
                # Gather the fitnesses of the previous generation, and use numpy
                #  to create a list sorting the element positions by fitness
                all_fitnesses_prev_gen = []
                for i in range(size_prev_gen):
                    all_fitnesses_prev_gen.append(sim_population.get_nn_fitness(i))
                self.assertEqual(len(all_fitnesses_prev_gen), size_prev_gen)
                
                all_fitnesses_prev_gen = np.asarray(all_fitnesses_prev_gen)
                most_fit_networks = np.argsort(all_fitnesses_prev_gen)
                most_fit_networks_desc = most_fit_networks[::-1].tolist()

                # Then iterate through the sorted list to copy the specified
                #  count of fit networks to the new generation, unless they have
                #  a minimum fitness.
                for i in range(carryover_count):
                    if sim_population.get_nn_fitness(most_fit_networks_desc[i]) != 1:
                        import_nn_definition = {}
                        import_nn_definition = sim_population.get_neural_network_def(most_fit_networks_desc[i])
                        self.assertIsInstance(import_nn_definition, dict)                    
                        # import_nn_weights_bias = population.Population.get_weight_bias_definitions(sim_population, most_fit_networks_desc[i], 1)
                        # import_nn_output_weights_bias = population.Population.get_weight_bias_definitions(sim_population, most_fit_networks_desc[i], 2)
                        import_nn_weights_bias = sim_population.get_weight_bias_definitions(most_fit_networks_desc[i], 1)
                        import_nn_output_weights_bias = sim_population.get_weight_bias_definitions(most_fit_networks_desc[i], 2)
                        import_nn_serial = import_nn_definition['meta']['serial_number']
                        import_nn_parent_1 = import_nn_definition['meta']['parent_1']
                        import_nn_parent_2 = import_nn_definition['meta']['parent_2']
                        self.assertIsInstance(import_nn_weights_bias, list)
                        self.assertIsInstance(import_nn_output_weights_bias, list)
                        
                        ############################
                        ### this is wrong. should not be using the sim_population obecjt.  Also see the child_nn_obj creation around line 356.
                        import_nn_obj = sim_population.create_nn(import_nn_serial, import_nn_definition, import_nn_weights_bias, import_nn_output_weights_bias, import_nn_parent_1, import_nn_parent_2)
                        new_population.add_nn(import_nn_obj)
                           
                       
            # print(f"\nFinal view of the population.")
            for i in range(new_population.get_population_size()):
                # print(f"{i}: {type(new_population.get_neural_network_model(i))}. ", end = "")
                # print(f"Serial Number: {population.Population.get_neural_network_def(new_population, i)['meta']['serial_number']}. ", end = "")
                # print(f"Checksum: {population.Population.get_neural_network_def(new_population, i)['meta']['checksum']}. ", end = "")
                
                # Let's make sure that any network with the same serial number in the old pop and the new also has the same checksum. Really they must.
                
                # for new_nn in range(population.Population.get_population_size(new_population)):
                for new_nn in range(new_population.get_population_size()):
                    # new_nn_sn = population.Population.get_neural_network_def(new_population, new_nn)['meta']['serial_number']
                    new_nn_sn = new_population.get_neural_network_def(new_nn)['meta']['serial_number']
                    # for old_nn in range(population.Population.get_population_size(sim_population)):
                    for old_nn in range(sim_population.get_population_size()):
                        old_nn_sn = population.Population.get_neural_network_def(sim_population, old_nn)['meta']['serial_number']
                        if old_nn_sn == new_nn_sn:
                            # new_nn_checksum = population.Population.get_neural_network_def(new_population, new_nn)['meta']['checksum']
                            # old_nn_checksum = population.Population.get_neural_network_def(sim_population, old_nn)['meta']['checksum']
                            new_nn_checksum = new_population.get_neural_network_def(new_nn)['meta']['checksum']
                            old_nn_checksum = sim_population.get_neural_network_def(old_nn)['meta']['checksum']
                            # print(f"Matching SN {old_nn_sn} == {new_nn_sn}. ", end = "")
                            # print(f"Checksum old {old_nn_checksum}, checksum new {new_nn_checksum}. ")
                            self.assertEqual(new_nn_checksum, old_nn_checksum)

                # There is no expectation here that the new objects will have a fitness since it hasn't been extracted from the old nn's and moved like the weights have been, and why would I? It's not useful anymore.
                
            
            # Replace the sim_population obj with the new_population obj
            sim_population = deepcopy(new_population)
            new_population = None
            self.assertIsNone(new_population)
            
            # Capture the state of the population with fitnesses after they've gone through the breeding and carryover
            experiment_results['after_carryover'] = reporting.Reporting.census(sim_population)
            experiment_report['generations'].append(experiment_results)
            
            print("Generation is complete.")    
        
        
        # print(f"Printing report: \n\n")
        # # print(f"Experiment parameters: {experiment_report['parameters']}\n\n")
        # print(f"Initial population (list of lists [meta, fitness]):")
        # for nn in range(len(experiment_report['initial_population'])):
        #     print(f"{nn}: {experiment_report['initial_population'][nn]}\n")
            
        # print(f"Now here is the stuff you need to work out. First, all of the 'generations' key.\n\n")
        # print(f"{experiment_report['generations']}\n\n")
        
        # for gen in range(len(experiment_report['generations'])):
        #     print(f"\nGeneration {gen}.")
        #     # print(f"\nPopulation after evaluation: {experiment_report['generations'][gen]['after_evaluation']}\n")
        #     print(f"\nPopulation after evaluation:")
        #     for nn in range(len(experiment_report['generations'][gen]['after_evaluation'])):
        #         print(f"{nn}: {experiment_report['generations'][gen]['after_evaluation'][nn]}\n")
        #     # print(f"\nPopulation after carryover: {experiment_report['generations'][gen]['after_carryover']}\n")
        #     print(f"\nPopulation after carryover:")
        #     for nn in range(len(experiment_report['generations'][gen]['after_carryover'])):
        #         print(f"{nn}: {experiment_report['generations'][gen]['after_carryover'][nn]}\n")
                
        # Export the report to disk
        reporting.Reporting.output_to_csv(experiment_number, experiment_report)
            
        
        # print(f"This is the before")
        # for nn in range(len(experiment_report['generations'][0])):
        #     print(f"{nn}: {experiment_report['generations'][0][nn]}\n")
        #     print(f"{nn}: {experiment_report['generations'][1][nn]}\n")
        #     print(f"\n")
            
        # for gen in range(len(experiment_report["generations"])):
        #     print(f"\nGeneration {gen}, before \n")
        #     print(experiment_report['generations'][gen])
        #     print(f"\nAfter evaluation:")
        #     print(experiment_report['generations'][gen][0])
        #     # for nn in range(len(experiment_report['generations'][gen][0])):
        #     #     print(f"{experiment_report['generations'][gen][0][nn]}")
        #     print(f"\nAfter breeding, mutation, and carryover:")
        #     print(experiment_report['generations'][gen][1])
        #     # for nn in range(len(experiment_report['generations'][gen][1])):
        #     #     print(f"{experiment_report['generations'][gen][1][nn]}")
        #     print("\n\n") 
        
        
        
        # Now here, all the generations have run, and the sim_population obj is finalised.
        # print(type(sim_population))
        
        
        
        
            # There is a microscopically small chance that the child definition is still the same as parent_1, or somehow all replaced with parent_2.
            # Smaller than 0.5^(# of neurons) small.  Really, really small.  Nevertheless, let's make sure.
            # self.assertNotEqual(child_nn_definition, parent_1_nn_definition)
            # self.assertNotEqual(child_nn_definition, parent_2_nn_definition)
                
                
                
            # Once the floats are fiddled with you can save the network object somehow. To a new population object?
            # If so then you'll need to remove the random initialisation of networks from population and make that callable instead
                
                
                
                # print("child network weights")
                # print((child_network_weights[0]))
                # child_network_weights_list = child_network_weights[0].tolist()
                # print(child_network_weights_list[0])
                # self.assertIsInstance(child_network_weights, list)
                
                # print(len(child_network_weights[1]))
                # # Crossover the parent hidden layer weights
                # for i in range(len(child_network_weights[1])):
                #     rand = random.random()
                #     if rand > 0.5:
                #         child_network_weights[1][i] = parent_2_nn_weights[1][i]
                #         print(child_network_weights[1][i], parent_1_nn_weights[1][i], parent_2_nn_weights[1][i])
                
                # child_network_weights_array = np.array(child_network_weights_list[0])
                # print(child_network_weights_array)
                
                
                # This is close. Check that codespeedy.com article. It's weights and biases, you need both.
                
                # Build the child network
                # The input layer must always be the same, so retrieve the definition from parent_1
                
                # Each parent may have a different number of layers. Crossover the same layer in each parent.
                # For extra layers, 50/50 to keep it and if so, copy as is
                # Each neuron is crossed with its counterpart on the other parent, 50/50 chance of either being selected
                # If there are extra neurons then it's 50/50 as to if it is retained. ###### implement this only if you chose to mutate the number of neurons
                
                # Mutate each weight, if chosen by randomness
                
                # Add or remove each hidden layer, if chosen by randomness
        
            
            
            # Remove the old population and set the new population
            
            
            # End of loop, once per generation
            
        return
            

            
        # Compare the elite networks and do stats and stuff.
            
            
            
# Run
        
unittest.main()


                # print(child_network_weights)
                # print(child_network_weights[0, 511])
                # child_network_weights[0, 511] = 1
                # print(child_network_weights)
                # print(child_network_weights[0, 511])