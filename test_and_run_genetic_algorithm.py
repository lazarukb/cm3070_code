import unittest
# import genome
import population
import simulation_utest
from copy import deepcopy
import random
import numpy as np
import reporting
from operator import itemgetter

# To remove unittest, if eventually decided ...
# Make sure the files are open (maybe, but just to be sure)
# Remove the unittest.TestCase argument and unittest.main() (if exists) calls from the module
# Rename the filename of the class module file

class TestGeneticAlgorithm(unittest.TestCase):
    
    # def __init__(self):
    
    # Code derived, but also fully re-typed and edited, from that presented in lectures in CM3020
    def testGeneticAlgorithm(self):
        # Define the population, simulation, and evolution parameters
        experiment_number = 307
        # This is also written as a folder in the experiment folder for quick 
        #   reference, so keep it short.
        experiment_comment = "fixed step scores 1 - each new generation should retain all of the original"
        serial_number = 307
        populations = []             ###### for possible future use in saving multiple populations for comparisons
        generations = 100
        size_new_generations = 20
        max_population_size = 40
        point_mutation_chance = 0.3
        point_mutation_amount = 0.35
        shrink_mutation_chance = 0.25
        growth_mutation_chance = 0.25
        force_random_choice = False
        force_pickup = False
        experiment_report = {}
        experiment_results = []
        game = 'coin_collector_5'
        
        # helpers_utest.TestHelpers.testVerifyParameters(self)
        
        # Early test of the directory structure to ensure nothing is overwritten.
        reporting.Reporting.create_folders(experiment_number, experiment_comment)
        
        # Create the initial population of randomised neural network definitions
        sim_population = population.Population()
        sim_population.test_create_random_population(size_new_generations, serial_number)
        serial_number += size_new_generations
        
        # Validate that the simulation_population is of the proper class, and has the specified number of member neural networks
        self.assertIsInstance(sim_population, population.Population)
        self.assertEqual(population.Population.get_population_size(sim_population), size_new_generations)
        
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
                       
            # Create the population instance target for the children
            new_population = population.Population()
            # Validate that the simulation_population is of the proper class, and has the specified number of member neural networks
            self.assertIsInstance(sim_population, population.Population)
            
            # run the networks through the game, which modifies the components of the simulation_population object
            ############################## WAIT, WHAT IS THIS MAX STEPS?
            sim_environment.evaluate_population(sim_population, game, force_random_choice, force_pickup)
            
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
            print(f"Proceed with breeding ", end = "")
            for breeding in range(size_new_generations):
                # Clear the child_nn_definition dict
                child_nn_definition = {}

                # Select two parent networks from the population, until two separate parents are chosen
                parent_1 = population.Population.get_weighted_parent(sim_population)
                self.assertIsInstance(parent_1, int)
                parent_2 = parent_1
                while parent_1 == parent_2:
                    parent_2 = population.Population.get_weighted_parent(sim_population)
                    self.assertIsInstance(parent_2, int)
                self.assertNotEqual(parent_1, parent_2)
                
                # print(f"\nParents: {parent_1}, {parent_2}")
                # print(f"Parent definitions: \n{population.Population.get_neural_network_def(sim_population, parent_1)}, \n{population.Population.get_neural_network_def(sim_population, parent_2)}")
            
                # Get the parental network definitions for future use               
                parent_1_nn_definition = population.Population.get_neural_network_def(sim_population, parent_1)
                parent_2_nn_definition = population.Population.get_neural_network_def(sim_population, parent_2)
                self.assertIsInstance(parent_1_nn_definition, dict)
                self.assertIsInstance(parent_2_nn_definition, dict)
                
                # Get the parental hidden layers element, for now assuming there is only one hidden layer                
                parent_1_nn_weights_bias = population.Population.get_weight_bias_definitions(sim_population, parent_1, 1)
                parent_2_nn_weights_bias = population.Population.get_weight_bias_definitions(sim_population, parent_2, 1)
                self.assertIsInstance(parent_1_nn_weights_bias, list)
                self.assertIsInstance(parent_2_nn_weights_bias, list)
                
                # Because the grandparents for both parents may be the same, there is a pretty good chance that at least some of the weights
                # in both parents are the same, inherited from the same grandparent. So we can no longer test all the weights to ensure they
                # changed, we have to test the checksum of the weights and activation and type floats all added. There is almost no chance
                # those are not unique between two networks, when measured to ~17 decimal places.
                self.assertNotEqual(parent_1_nn_definition["meta"]["checksum"], parent_2_nn_definition["meta"]["checksum"])
                
                
                # print("Children crossover and mutation stuff")
                # Create a child neural network from the parents, and potentially affected by mutation.
                
                # Create the child hidden layer weights as a full copy of parent 1. Then as necessary, replace with values from parent 2, thus achieving crossover breeding.
                child_network_weights_bias = deepcopy(parent_1_nn_weights_bias)
                child_network_weights = child_network_weights_bias[0]       ##### currently assuming that there is only one hidden layer
                # Randomly sample to make sure the copy worked
                # rand = random.randint(0, len(parent_1_nn_weights_bias[0][0]) - 1)
                # self.assertEqual(child_network_weights[0][rand], parent_1_nn_weights_bias[0][0][rand])
                
                # Assert checks that the parents are fully different because their weight checksums are
                self.assertNotEqual(parent_1_nn_definition["meta"]["checksum"], parent_2_nn_definition["meta"]["checksum"])
                
                # Iterate through the child hidden layer weights, applying crossover and mutation to each
                completed_iterations = 0
                for i in range (len(child_network_weights[0])):
                    # Crossover, 50% chance for each parent, for each neuron
                    rand = random.random() + 0 ###############################################################
                    if rand < 0.5:
                        child_network_weights[0][i] = parent_2_nn_weights_bias[0][0][i]
                        # self.assertNotEqual(parent_1_nn_weights_bias[0][0][i], parent_2_nn_weights_bias[0][0][i])
                        # self.assertNotEqual(child_network_weights[0][i], parent_1_nn_weights_bias[0][0][i])

                    # Point mutation change, for each neuron
                    rand = random.random() + 0 ###############################################################
                    if rand < point_mutation_chance:
                        # Randomly select an increase or decrease amount
                        mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
                        child_network_weights[0][i] += mutation_amount
                        if child_network_weights[0][i] < -1.0:
                            child_network_weights[0][i] = -1.0
                        elif child_network_weights[0][i] > 1.0:
                            child_network_weights[0][i] = 1.0
                    completed_iterations += 1
                    
                    # Make sure the neuron weights are within the acceptable bounds, whether or not they were mutated.
                    self.assertGreaterEqual(child_network_weights[0][i], -1.0)
                    self.assertLessEqual(child_network_weights[0][i], 1.0)
                
                # Make sure we went over all the neurons
                self.assertEqual(completed_iterations, len(child_network_weights[0]))
                
                # Write the new weights back to the weights+biases variable, for later writing to the network object.            
                child_network_weights_bias[0] = child_network_weights
                
                # Now crossover and mutate the floats in the DNA
                # Again start with the child being a full copy of parent 1, then achieve crossover by replacing with parts of parent_2.
                child_nn_definition = deepcopy(parent_1_nn_definition)
                
                # Nothing in the input layer to crossover or mutate. 
                # If we change the inputs then the network will not be able to interact with OpenAI gym and its fitness will be 0, so it will "die". 
                # No point in going through the effort to mutate a network that will use be useless.
                
                # print("Output of the parent's definitions")
                # print(parent_1_nn_definition)
                # print(parent_2_nn_definition)
                
                # Massive duplication in here - move this to a helper function
                
                # Iterate through the child hidden layer floats in the DNA, applying crossover and mutation.
                for layer in range(len(child_nn_definition["hidden_layers"])):
                    # Crossover, 50% chance for each parent, for each type and activation
                    rand = random.random() + 0 ###############################################################
                    if rand < 0.5:
                        child_nn_definition["hidden_layers"][layer]["type"] = parent_2_nn_definition["hidden_layers"][layer]["type"]
                        # self.assertNotEqual(parent_1_nn_definition["hidden_layers"][layer]["type"], parent_2_nn_definition["hidden_layers"][layer]["type"])
                        # self.assertNotEqual(child_nn_definition["hidden_layers"][layer]["type"], parent_1_nn_definition["hidden_layers"][layer]["type"])
                    
                    rand = random.random() + 0 ###############################################################
                    if rand < 0.5:
                        child_nn_definition["hidden_layers"][layer]["activation"] = parent_2_nn_definition["hidden_layers"][layer]["activation"]
                        
                    # Point mutation, for each type and activation
                    rand = random.random() + 0 ###############################################################
                    if rand < point_mutation_chance:
                        # Randomly select an increase or decrease amount
                        mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
                        child_nn_definition["hidden_layers"][layer]["type"] += mutation_amount
                        if child_nn_definition["hidden_layers"][layer]["type"] < 0.0: 
                            child_nn_definition["hidden_layers"][layer]["type"] = 0.0
                        elif child_nn_definition["hidden_layers"][layer]["type"] > 1.0:
                            child_nn_definition["hidden_layers"][layer]["type"] = 1.0
    
                    rand = random.random() + 0 ###############################################################
                    if rand < point_mutation_chance:
                        # Randomly select an increase or decrease amount
                        mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
                        child_nn_definition["hidden_layers"][layer]["activation"] += mutation_amount
                        if child_nn_definition["hidden_layers"][layer]["activation"] < 0.0:
                            child_nn_definition["hidden_layers"][layer]["activation"] = 0.0
                        elif child_nn_definition["hidden_layers"][layer]["activation"] > 1.0:
                            child_nn_definition["hidden_layers"][layer]["activation"] = 1.0
                    
                    # Validate that all the floats are within the acceptable ranges.
                    # print(child_nn_definition["hidden_layers"][layer]["activation"])
                    self.assertGreaterEqual(child_nn_definition["hidden_layers"][layer]["type"], 0.0)
                    self.assertLessEqual(child_nn_definition["hidden_layers"][layer]["type"], 1.0)
                    self.assertGreaterEqual(child_nn_definition["hidden_layers"][layer]["activation"], 0.0)
                    self.assertLessEqual(child_nn_definition["hidden_layers"][layer]["activation"], 1.0)
                    
                
                # Repeat for the output layer - remember to move all this to a helper function or three.
                # Crossover, 50% chance for each parent, for each type and activation
                rand = random.random() + 0 ###############################################################
                if rand < 0.5:
                    child_nn_definition["output"]["type"] = parent_2_nn_definition["output"]["type"]
                    # self.assertNotEqual(parent_1_nn_definition["output"]["type"], parent_2_nn_definition["output"]["type"])
                    # self.assertNotEqual(child_nn_definition["output"]["type"], parent_1_nn_definition["output"]["type"])
                
                rand = random.random() + 0 ###############################################################
                if rand < 0.5:
                    child_nn_definition["output"]["activation"] = parent_2_nn_definition["output"]["activation"]
                    # self.assertNotEqual(parent_1_nn_definition["output"]["activation"], parent_2_nn_definition["output"]["activation"])
                    # self.assertNotEqual(child_nn_definition["output"]["activation"], parent_1_nn_definition["output"]["activation"])
                    
                # Point mutation, for each type and activation
                rand = random.random() + 0 ###############################################################
                if rand < point_mutation_chance:
                    # Randomly select an increase or decrease amount
                    mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
                    child_nn_definition["output"]["type"] += mutation_amount
                    if child_nn_definition["output"]["type"] < 0.0: 
                        child_nn_definition["output"]["type"] = 0.0
                    elif child_nn_definition["output"]["type"] > 1.0:
                        child_nn_definition["output"]["type"] = 1.0

                rand = random.random() + 0 ###############################################################
                if rand < point_mutation_chance:
                    # Randomly select an increase or decrease amount
                    mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
                    child_nn_definition["output"]["activation"] += mutation_amount
                    if child_nn_definition["output"]["activation"] < 0.0: 
                        child_nn_definition["output"]["activation"] = 0.0
                    elif child_nn_definition["output"]["activation"] > 1.0:
                        child_nn_definition["output"]["activation"] = 1.0
                        
                # Validate that all the floats are within the acceptable ranges.
                # print(child_nn_definition["output"]["type"])
                self.assertGreaterEqual(child_nn_definition["output"]["type"], 0.0)
                self.assertLessEqual(child_nn_definition["output"]["type"], 1.0)
                self.assertGreaterEqual(child_nn_definition["output"]["activation"], 0.0)
                self.assertLessEqual(child_nn_definition["output"]["activation"], 1.0)                   

                # At this point:
                # The code above is still assuming that only one hidden layer exists. **************************
                # All the weights of the hidden layer are crossed over and mutated.
                # All the floats of the hidden layer are crossed over and mutated.
                # All the weights of the output layer are **** NOT **** crossed over and mutated. ****************
                # All the floats of the output layer are crossed over and mutated.
                # print("Output of the child definition")
                # print(child_nn_definition)
                
                
                # This will retrieve the last elements of the layers information, which will be the output layer
                # Continues to assume there is only one hidden layer which needs to be changed later on.
                # This layer also has 512 weights #####################################################################################
                parent_1_nn_output_weights_bias = population.Population.get_weight_bias_definitions(sim_population, parent_1, 2)
                parent_2_nn_output_weights_bias = population.Population.get_weight_bias_definitions(sim_population, parent_2, 2)
                

                # print("Let's look at the weights and biases information for the output layer")
                # print(parent_1_nn_output_weights_bias)
                # print(len(parent_1_nn_output_weights_bias[0]))
                
                
                # Create the child output layer weights as a full copy of parent 1. Then if necessary, replace with values from parent 2, thus achieving crossover breeding.
                child_network_output_weights_bias = deepcopy(parent_1_nn_output_weights_bias)
                child_network_output_weights = child_network_output_weights_bias[0]       ##### currently assuming that there is only one hidden layer
                child_network_output_weights_initial = deepcopy(child_network_output_weights)
                # Randomly sample to make sure the copy worked
                rand = random.randint(0, len(parent_1_nn_output_weights_bias[0][0]) - 1)
                self.assertEqual(child_network_output_weights[0][rand], parent_1_nn_output_weights_bias[0][0][rand])
                
                # Iterate through the child output layer weights, applying crossover and mutation to each
                completed_iterations = 0
                for i in range (len(child_network_output_weights[0])):
                    # Crossover, 50% chance for each parent, for each neuron
                    rand = random.random() + 0 ###############################################################
                    if rand < 0.5:
                        child_network_output_weights[0][i] = parent_2_nn_output_weights_bias[0][0][i]
                        # self.assertNotEqual(parent_1_nn_output_weights_bias[0][0][i], parent_2_nn_output_weights_bias[0][0][i])
                        # self.assertNotEqual(child_network_output_weights[0][i], parent_1_nn_output_weights_bias[0][0][i])

                    # Point mutation change, for each neuron
                    rand = random.random() + 0 ###############################################################
                    if rand < point_mutation_chance:
                        # Randomly select an increase or decrease amount
                        mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
                        child_network_output_weights[0][i] += mutation_amount
                        if child_network_output_weights[0][i] < -1.0: 
                            child_network_output_weights[0][i] = -1.0
                        elif child_network_output_weights[0][i] > 1.0:
                            child_network_output_weights[0][i] = 1.0
                    completed_iterations += 1
                

                    # Make sure the neuron weights are within the acceptable bounds, whether or not they were mutated
                    self.assertGreaterEqual(child_network_output_weights[0][i], -1.0)
                    self.assertLessEqual(child_network_output_weights[0][i], 1.0)
                    
                # Make sure we went over all the neurons
                self.assertEqual(completed_iterations, len(child_network_output_weights[0]))
                
                # Write the new weights back to the weights+biases variable, for later writing to the network object.            
                child_network_output_weights_bias[0] = child_network_output_weights
                
                # Here we should be ready to create the new network definition dict and weights
                # Child network dict: child_nn_definition
                # Child network hidden weights+biases: child_network_weights_bias
                # Child network output weights+biases: child_network_output_weights_bias
                # print("Child being created")
                # print(child_nn_definition)
                # print(child_network_weights_bias)
                # print(child_network_output_weights_bias)
                # child_nn_obj = population.Population.create_nn(self, serial_number, child_nn_definition, child_network_weights_bias, child_network_output_weights_bias, parent_1, parent_2)
                child_nn_obj = sim_population.create_nn(serial_number, child_nn_definition, child_network_weights_bias, child_network_output_weights_bias, parent_1, parent_2)
                
                # Do some assertions to confirm that the child nn is properly defined and populated
                # self.assertIsInstance(child_nn_obj.get_network_dna()["meta"]["serial_number"], serial_number)
                # print("DNA from the created nn obj")
                # print(child_nn_obj.get_network_dna())
                self.assertIsInstance(child_nn_obj.get_network_dna()["input"], int)
                self.assertIsNot(len(child_nn_obj.get_network_dna()["hidden_layers"]), 0)
                self.assertIsInstance(child_nn_obj.get_network_dna()["output"], dict)
                self.assertIsNot(child_nn_obj.checksum_weights(1), 0)
                self.assertIsNot(child_nn_obj.checksum_weights(2), 0)
                
                # And add it to the new population
                # print(f"Add child to the new population")
                new_population.add_nn(child_nn_obj)
                serial_number += 1
                
                # print("Let's look at the crossed and mutated weights and biases information for the output layer. To see if something has in fact changed.")
                # print(child_network_output_weights_bias)
                # print(len(child_network_output_weights_bias[0]))   
            # else:
            #     # Not enough individuals with positive fitness. Throw out the batch
            #     print(f"Not enough individuals with positive fitness. Throw out the batch. ")
            #     print(f"Fitness map: {fitness_map}")
            #     new_population = population.Population()
            #     new_population.test_create_random_population(init_population_size, serial_number)

            # print("\nBreeding cycle is complete. Moving on to cleanup.")
            # That should be it. The new_population should have a population count of init_population_size
            # self.assertEqual(population.Population.get_population_size(new_population), size_new_generations)
            

            
            
            
            # If you're going to retain the most fit individual(s) from the previous generation, here is where that should happen
            # select the most fit individuals from the existing generation and move them into the new generation
            if size_new_generations != max_population_size:
                # Get the number of networks to carry over from the old generation to the new
                # But cap at the number of networks in the old generation.
                
                print(f"\nProduce carryover from the old generation to the new. ", end = "")
                size_prev_gen = population.Population.get_population_size(sim_population)
                carryover_count = max_population_size - size_new_generations
                carryover_count = min(carryover_count, size_prev_gen)
                print(f"{carryover_count} networks from the old generation will import to the new generation.")
                # carryover_count = 1
                
                all_fitnesses_prev_gen = []
                for i in range(size_prev_gen):
                    all_fitnesses_prev_gen.append(sim_population.get_nn_fitness(i))
                self.assertEqual(len(all_fitnesses_prev_gen), size_prev_gen)
                
                all_fitnesses_prev_gen = np.asarray(all_fitnesses_prev_gen)
                
                most_fit_networks = np.argsort(all_fitnesses_prev_gen)
                most_fit_networks_desc = most_fit_networks[::-1].tolist()
                # print(f"{all_fitnesses_prev_gen}")
                # print(f"{most_fit_networks}, {type(most_fit_networks)}")
                # print(f"{most_fit_networks_desc}, {type(most_fit_networks_desc)}")
                
                for i in range(carryover_count):
                    # print(f"This is iteration {i}.")
                    # print(f"The index number we are going to copy is {most_fit_networks_desc[i]}. ", end = "")
                    
                    # This should be largely the same as creating a child network.
                    # Get the network
                    # nn_to_import = population.Population.get_neural_network_model(sim_population, most_fit_networks_desc[i])
                    # nn_to_import = population.Population.get_neural_network_def(sim_population, i)
                    
                    import_nn_definition = {}
                    import_nn_definition = population.Population.get_neural_network_def(sim_population, most_fit_networks_desc[i])
                    self.assertIsInstance(import_nn_definition, dict)
                    # print(f"Serial number of network index {i} is {import_nn_definition['meta']['serial_number']}.")
                    # print(f"Checksum of network index {i} is {import_nn_definition['meta']['checksum']}.")
                    
                    import_nn_weights_bias = population.Population.get_weight_bias_definitions(sim_population, most_fit_networks_desc[i], 1)
                    import_nn_output_weights_bias = population.Population.get_weight_bias_definitions(sim_population, most_fit_networks_desc[i], 2)
                    import_nn_serial = import_nn_definition['meta']['serial_number']
                    import_nn_parent_1 = import_nn_definition['meta']['parent_1']
                    import_nn_parent_2 = import_nn_definition['meta']['parent_2']
                    
                    self.assertIsInstance(import_nn_weights_bias, list)
                    self.assertIsInstance(import_nn_output_weights_bias, list)
                    
                    # print(f"import_nn_serial: {import_nn_serial}, import_nn_parent_1: {import_nn_parent_1}, import_nn_parent_2: {import_nn_parent_2}")
                    
                    ############################
                    ### this is wrong. should not be using the sim_population obecjt.  Also see the child_nn_obj creation around line 356.
                    import_nn_obj = sim_population.create_nn(import_nn_serial, import_nn_definition, import_nn_weights_bias, import_nn_output_weights_bias, import_nn_parent_1, import_nn_parent_2)
                    new_population.add_nn(import_nn_obj)
                    
                    # print(f"{import_nn_definition}")
                    
                    # import_nn_definition = population.Population.get_neural_network_def(sim_population, parent_1)         
                    # import_nn_weights_bias = population.Population.get_weight_bias_definitions(sim_population, parent_1, 1)
                    # import_nn_output_weights_bias = population.Population.get_weight_bias_definitions(sim_population, parent_1, 2)
                    # self.assertIsInstance(parent_1_nn_definition, dict)
                    
                    # Still need to extract the original parent 1, parent 2, and serial number for this to work
                    # import_sn = 
                    # import_parent_1 = 
                    # import_parent_2 = 
                    
                    # import_nn_obj = population.Population.create_nn(self, serial_number, import_nn_definition, import_nn_weights_bias, import_nn_output_weights_bias, parent_1, parent_2)
                    # new_population.add_nn(import_nn_obj)
                    
                    # self.assertIsInstance(nn_to_import.get_network_dna()["input"], int)
                    # self.assertIsNot(len(nn_to_import.get_network_dna()["hidden_layers"]), 0)
                    # self.assertIsInstance(nn_to_import.get_network_dna()["output"], dict)
                    
                    # quit()
                    
                # self.assertEqual(population.Population.get_population_size(new_population), size_new_generations + carryover_count)
                
                # fitness_map = sim_population.get_fitness_map()
                # Get the elements of the fitness_map sorted by fitness descending so we can easily grab the x most fit networks
                
                # for i in range(carryover_count):
                    
                #     network_to_import = all_fitnesses_prev_gen
                #     # assertion
                #     if i == 0:
                #         self.assertEqual(fitness_map[::-i + 1], fitness_map[len(fitness_map) - 1])
                #         quit()
                           
                       
            # print(f"\nFinal view of the population.")
            for i in range(new_population.get_population_size(new_population)):
                # print(f"{i}: {type(new_population.get_neural_network_model(i))}. ", end = "")
                # print(f"Serial Number: {population.Population.get_neural_network_def(new_population, i)['meta']['serial_number']}. ", end = "")
                # print(f"Checksum: {population.Population.get_neural_network_def(new_population, i)['meta']['checksum']}. ", end = "")
                
                # Let's make sure that any network with the same serial number in the old pop and the new also has the same checksum. Really they must.
                
                for new_nn in range(population.Population.get_population_size(new_population)):
                    new_nn_sn = population.Population.get_neural_network_def(new_population, new_nn)['meta']['serial_number']
                    for old_nn in range(population.Population.get_population_size(sim_population)):
                        old_nn_sn = population.Population.get_neural_network_def(sim_population, old_nn)['meta']['serial_number']
                        if old_nn_sn == new_nn_sn:
                            new_nn_checksum = population.Population.get_neural_network_def(new_population, new_nn)['meta']['checksum']
                            old_nn_checksum = population.Population.get_neural_network_def(sim_population, old_nn)['meta']['checksum']
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
            
            
            
            
        
unittest.main()


                # print(child_network_weights)
                # print(child_network_weights[0, 511])
                # child_network_weights[0, 511] = 1
                # print(child_network_weights)
                # print(child_network_weights[0, 511])