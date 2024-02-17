import population
import unittest
from copy import deepcopy
import random
import numpy as np


class Breeding(unittest.TestCase):
    
    def cross_and_mutate(self, sim_population, new_population, serial_number, point_mutation_scalar, point_mutation_chance, point_mutation_amount, point_mutation_chance_max, point_mutation_amount_max):
        # Clear the child_nn_definition dict
        child_nn_definition = {}

        # Select two parent networks from the population, until two unique parents are chosen
        parent_1 = population.Population.get_weighted_parent(sim_population)
        self.assertIsInstance(parent_1, int)
        parent_2 = parent_1
        while parent_1 == parent_2:
            parent_2 = population.Population.get_weighted_parent(sim_population)
            self.assertIsInstance(parent_2, int)
        self.assertNotEqual(parent_1, parent_2)
        
        
        # If both parents have fitness of 1 then neither succeeded at the game
        # Increase the mutation chance and factor, to further distance the child from the parent definitions.
        parent_1_fitness = sim_population.get_nn_fitness(parent_1)
        parent_2_fitness = sim_population.get_nn_fitness(parent_2)
        if (parent_1_fitness == 1) and (parent_2_fitness == 1):
            self.assertEqual(parent_1_fitness, parent_2_fitness)
            # print(f"{point_mutation_chance}, {point_mutation_amount}", end = "")
            point_mutation_chance *= point_mutation_scalar
            point_mutation_amount *= point_mutation_scalar
            point_mutation_chance = min(point_mutation_chance, point_mutation_chance_max)
            point_mutation_amount = min(point_mutation_amount, point_mutation_amount_max)
            # print(f" --> {point_mutation_chance}, {point_mutation_amount}")

        # Get the parental network definitions for future use
        parent_1_nn_definition = sim_population.get_neural_network_def(parent_1)
        parent_2_nn_definition = sim_population.get_neural_network_def(parent_2)
        self.assertIsInstance(parent_1_nn_definition, dict)
        self.assertIsInstance(parent_2_nn_definition, dict)
        
        # Get the parental hidden layers element, for now assuming there is only one hidden layer                
        parent_1_nn_weights_bias = sim_population.get_weight_bias_definitions(parent_1, 1)
        parent_2_nn_weights_bias = sim_population.get_weight_bias_definitions(parent_2, 1)
        self.assertIsInstance(parent_1_nn_weights_bias, list)
        self.assertIsInstance(parent_2_nn_weights_bias, list)
        
        # Because the grandparents for both parents may be the same, there is a pretty good chance that at least some of the weights
        # in both parents are the same, inherited from the same grandparent. So we can no longer test all the weights to ensure they
        # changed, we have to test the checksum of the weights and activation and type floats all added. There is almost no chance
        # those are not unique between two networks, when measured to ~17 decimal places.
        self.assertNotEqual(parent_1_nn_definition["meta"]["checksum"], parent_2_nn_definition["meta"]["checksum"])

        # Create a child neural network from the parents, and potentially affected by mutation.        
        # Create the child hidden layer weights as a full copy of parent 1. Then as necessary, replace with values from parent 2, thus achieving crossover breeding.
        child_network_weights_bias = deepcopy(parent_1_nn_weights_bias)
        child_network_weights = child_network_weights_bias[0]       ##### currently assuming that there is only one hidden layer

        # This assert checks that the parents are fully different 
        #  because their weight checksums are
        self.assertNotEqual(parent_1_nn_definition["meta"]["checksum"], parent_2_nn_definition["meta"]["checksum"])
        
        # Iterate through the child hidden layer weights, applying crossover and mutation to each
        completed_iterations = 0
        
        # pass child_network_weights[0] and parent_2_nn_weights_bias[0][0],
        #  and weighted chance of parent_2 being selected
        for i in range (len(child_network_weights[0])):
            # Crossover, 50% chance for each parent, for each neuron
            # rand = random.random() + 0 ###############################################################
            # if rand < 0.5:
            #     child_network_weights[0][i] = parent_2_nn_weights_bias[0][0][i]
                # self.assertNotEqual(parent_1_nn_weights_bias[0][0][i], parent_2_nn_weights_bias[0][0][i])
                # self.assertNotEqual(child_network_weights[0][i], parent_1_nn_weights_bias[0][0][i])
                
            child_network_weights[0][i] = self.c_and_m("weight", child_network_weights[0][i], parent_2_nn_weights_bias[0][0][i], point_mutation_chance, point_mutation_amount)

            # Point mutation change, for each neuron
            # rand = random.random() + 0 ###############################################################
            # if rand < point_mutation_chance:
            #     # Randomly select an increase or decrease amount
            #     mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
            #     child_network_weights[0][i] += mutation_amount
            #     if child_network_weights[0][i] < -1.0:
            #         child_network_weights[0][i] = -1.0
            #     elif child_network_weights[0][i] > 1.0:
            #         child_network_weights[0][i] = 1.0
            completed_iterations += 1
            
            # Make sure the neuron weights are within the acceptable bounds, whether or not they were mutated.
            self.assertIsInstance(child_network_weights[0][i], np.float32)
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
            # rand = random.random() + 0 ###############################################################
            # if rand < 0.5:
            #     child_nn_definition["hidden_layers"][layer]["type"] = parent_2_nn_definition["hidden_layers"][layer]["type"]
            #     # self.assertNotEqual(parent_1_nn_definition["hidden_layers"][layer]["type"], parent_2_nn_definition["hidden_layers"][layer]["type"])
            #     # self.assertNotEqual(child_nn_definition["hidden_layers"][layer]["type"], parent_1_nn_definition["hidden_layers"][layer]["type"])
            
            # rand = random.random() + 0 ###############################################################
            # if rand < 0.5:
            #     child_nn_definition["hidden_layers"][layer]["activation"] = parent_2_nn_definition["hidden_layers"][layer]["activation"]
                
            # # Point mutation, for each type and activation
            # rand = random.random() + 0 ###############################################################
            # if rand < point_mutation_chance:
            #     # Randomly select an increase or decrease amount
            #     mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
            #     child_nn_definition["hidden_layers"][layer]["type"] += mutation_amount
            #     if child_nn_definition["hidden_layers"][layer]["type"] < 0.0: 
            #         child_nn_definition["hidden_layers"][layer]["type"] = 0.0
            #     elif child_nn_definition["hidden_layers"][layer]["type"] > 1.0:
            #         child_nn_definition["hidden_layers"][layer]["type"] = 1.0

            # rand = random.random() + 0 ###############################################################
            # if rand < point_mutation_chance:
            #     # Randomly select an increase or decrease amount
            #     mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
            #     child_nn_definition["hidden_layers"][layer]["activation"] += mutation_amount
            #     if child_nn_definition["hidden_layers"][layer]["activation"] < 0.0:
            #         child_nn_definition["hidden_layers"][layer]["activation"] = 0.0
            #     elif child_nn_definition["hidden_layers"][layer]["activation"] > 1.0:
            #         child_nn_definition["hidden_layers"][layer]["activation"] = 1.0
                    
            child_nn_definition["hidden_layers"][layer]["activation"] = self.c_and_m("definition", child_nn_definition["hidden_layers"][layer]["activation"], parent_2_nn_definition["hidden_layers"][layer]["activation"], point_mutation_chance, point_mutation_amount)
            child_nn_definition["hidden_layers"][layer]["type"] = self.c_and_m("definition", child_nn_definition["hidden_layers"][layer]["type"], parent_2_nn_definition["hidden_layers"][layer]["type"], point_mutation_chance, point_mutation_amount)
            
            # Validate that all the floats are within the acceptable ranges.
            # print(child_nn_definition["hidden_layers"][layer]["activation"])
            self.assertIsInstance(child_nn_definition["hidden_layers"][layer]["type"], float)
            self.assertIsInstance(child_nn_definition["hidden_layers"][layer]["activation"], float)
            self.assertGreaterEqual(child_nn_definition["hidden_layers"][layer]["type"], 0.0)
            self.assertLessEqual(child_nn_definition["hidden_layers"][layer]["type"], 1.0)
            self.assertGreaterEqual(child_nn_definition["hidden_layers"][layer]["activation"], 0.0)
            self.assertLessEqual(child_nn_definition["hidden_layers"][layer]["activation"], 1.0)
            
        
        # Repeat for the output layer - remember to move all this to a helper function or three.
        # Crossover, 50% chance for each parent, for each type and activation
        # rand = random.random() + 0 ###############################################################
        # if rand < 0.5:
        #     child_nn_definition["output"]["type"] = parent_2_nn_definition["output"]["type"]
        #     # self.assertNotEqual(parent_1_nn_definition["output"]["type"], parent_2_nn_definition["output"]["type"])
        #     # self.assertNotEqual(child_nn_definition["output"]["type"], parent_1_nn_definition["output"]["type"])
            
        child_nn_definition["output"]["type"] = self.c_and_m("definition", child_nn_definition["output"]["type"], parent_2_nn_definition["output"]["type"], point_mutation_chance, point_mutation_amount)
        
        # rand = random.random() + 0 ###############################################################
        # if rand < 0.5:
        #     child_nn_definition["output"]["activation"] = parent_2_nn_definition["output"]["activation"]
        #     # self.assertNotEqual(parent_1_nn_definition["output"]["activation"], parent_2_nn_definition["output"]["activation"])
        #     # self.assertNotEqual(child_nn_definition["output"]["activation"], parent_1_nn_definition["output"]["activation"])
        
        child_nn_definition["output"]["activation"] = self.c_and_m("definition", child_nn_definition["output"]["activation"], parent_2_nn_definition["output"]["activation"], point_mutation_chance, point_mutation_amount)
            
        # Point mutation, for each type and activation
        # rand = random.random() + 0 ###############################################################
        # if rand < point_mutation_chance:
        #     # Randomly select an increase or decrease amount
        #     mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
        #     child_nn_definition["output"]["type"] += mutation_amount
        #     if child_nn_definition["output"]["type"] < 0.0: 
        #         child_nn_definition["output"]["type"] = 0.0
        #     elif child_nn_definition["output"]["type"] > 1.0:
        #         child_nn_definition["output"]["type"] = 1.0

        # rand = random.random() + 0 ###############################################################
        # if rand < point_mutation_chance:
        #     # Randomly select an increase or decrease amount
        #     mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
        #     child_nn_definition["output"]["activation"] += mutation_amount
        #     if child_nn_definition["output"]["activation"] < 0.0: 
        #         child_nn_definition["output"]["activation"] = 0.0
        #     elif child_nn_definition["output"]["activation"] > 1.0:
        #         child_nn_definition["output"]["activation"] = 1.0
                
        # Validate that all the floats are within the acceptable ranges.
        # print(child_nn_definition["output"]["type"])
        self.assertIsInstance(child_nn_definition["output"]["type"], float)
        self.assertIsInstance(child_nn_definition["output"]["activation"], float)
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
        parent_1_nn_output_weights_bias = sim_population.get_weight_bias_definitions(parent_1, 2)
        parent_2_nn_output_weights_bias = sim_population.get_weight_bias_definitions(parent_2, 2)
        

        # print("Let's look at the weights and biases information for the output layer")
        # print(parent_1_nn_output_weights_bias)
        # print(len(parent_1_nn_output_weights_bias[0]))
        
        
        # Create the child output layer weights as a full copy of parent 1. Then if necessary, replace with values from parent 2, thus achieving crossover breeding.
        child_network_output_weights_bias = deepcopy(parent_1_nn_output_weights_bias)
        child_network_output_weights = child_network_output_weights_bias[0]       ##### currently assuming that there is only one hidden layer
        child_network_output_weights_initial = deepcopy(child_network_output_weights)
        # Randomly sample to make sure the copy worked
        for i in range(10):
            rand = random.randint(0, len(parent_1_nn_output_weights_bias[0][0]) - 1)
            self.assertEqual(child_network_output_weights[0][rand], parent_1_nn_output_weights_bias[0][0][rand])
        
        # Iterate through the child output layer weights, applying crossover and mutation to each
        completed_iterations = 0
        for i in range (len(child_network_output_weights[0])):
            # Crossover, 50% chance for each parent, for each neuron
            child_network_output_weights[0][i] = self.c_and_m("weight", child_network_output_weights[0][i], parent_2_nn_output_weights_bias[0][0][i], point_mutation_chance, point_mutation_amount)


            # rand = random.random() + 0 ###############################################################
            # if rand < 0.5:
            #     child_network_output_weights[0][i] = parent_2_nn_output_weights_bias[0][0][i]
            #     # self.assertNotEqual(parent_1_nn_output_weights_bias[0][0][i], parent_2_nn_output_weights_bias[0][0][i])
            #     # self.assertNotEqual(child_network_output_weights[0][i], parent_1_nn_output_weights_bias[0][0][i])

            # # Point mutation change, for each neuron
            # rand = random.random() + 0 ###############################################################
            # if rand < point_mutation_chance:
            #     # Randomly select an increase or decrease amount
            #     mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
            #     child_network_output_weights[0][i] += mutation_amount
            #     if child_network_output_weights[0][i] < -1.0: 
            #         child_network_output_weights[0][i] = -1.0
            #     elif child_network_output_weights[0][i] > 1.0:
            #         child_network_output_weights[0][i] = 1.0
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
        
        # This was wrong. It was storing the number of the parent from the population object, not the serial number of the parent.
        # child_nn_obj = sim_population.create_nn(serial_number, child_nn_definition, child_network_weights_bias, child_network_output_weights_bias, parent_1, parent_2)
        
        parent_1_sn = parent_1_nn_definition['meta']['serial_number']
        parent_2_sn = parent_2_nn_definition['meta']['serial_number']
        
        # print(f"\n\np1 number: {parent_1}, p1_sn: {parent_1_sn}, p2 number: {parent_2}, p2_sn: {parent_2_sn}\n\n")
        child_nn_obj = sim_population.create_nn(serial_number, child_nn_definition, child_network_weights_bias, child_network_output_weights_bias, parent_1_sn, parent_2_sn)
        
        # Do some assertions to confirm that the child nn is properly defined and populated
        # self.assertIsInstance(child_nn_obj.get_network_dna()["meta"]["serial_number"], serial_number)
        # print("DNA from the created nn obj")
        # print(child_nn_obj.get_network_dna())
        self.assertIsInstance(child_nn_obj.get_network_dna()["inputs"], int)
        self.assertIsNot(len(child_nn_obj.get_network_dna()["hidden_layers"]), 0)
        self.assertIsInstance(child_nn_obj.get_network_dna()["output"], dict)
        self.assertIsNot(child_nn_obj.checksum_weights(1), 0)
        self.assertIsNot(child_nn_obj.checksum_weights(2), 0)
        
        # And add it to the new population
        new_population.add_nn(child_nn_obj)
        # serial_number += 1
    
        self.assertIsInstance(new_population, population.Population)
        return new_population
    
    def carry_over():
        pass
    
    def c_and_m(self, weight_or_def, child_value, parent_value, point_mutation_chance, point_mutation_amount, fitness_bias = 0):
        # Weights have a range from -1.0 to 1.0
        # Definitions have a range from 0.0 to 1.0
        if weight_or_def == "weight":
            range_min = -1.0
            range_max = 1.0
        elif weight_or_def == "definition":
            range_min = 0.0
            range_max = 1.0
        else:
            quit()
        
        # Choose either the child value, which is currently a copy of parent_1,
        #  or the parent_2 value
        rand = random.random()
        used_parent_2 = None
        threshold = 0.5 + fitness_bias
        if rand < threshold:
            result = parent_value
            used_parent_2 = True
        else:
            result = child_value
            used_parent_2 = False
        
        # Since used_parent_2 is only changed from None in the loops then
        #  it must be a Boolean iff the crossover was processed
        if rand < threshold:
            self.assertTrue(used_parent_2)
        else:
            self.assertFalse(used_parent_2)
        
        # Now mutate the selected value, or not.
        rand = random.random()
        mutated = None
        if rand < point_mutation_chance:
            mutated = True
            # Randomly select an increase or decrease amount
            mutation_amount = random.uniform(-point_mutation_amount, point_mutation_amount)
            result += mutation_amount
            if result < range_min:
                result = range_min
            elif result > range_max:
                result = range_max
        else:
            mutated = False
                
        # If mutation happened then the result value cannot identical to
        #  the child or parent value, depending on which was the seed.
        # Unless the result is 0.0 or 1.0 as the seed value could also have been
        #  so capped. In that case only the first assertion will have value,
        #  showing that the mutation was at minimum considered.
        self.assertIsNotNone(mutated)
        if mutated and result != 0 and result != 1.0:
            if used_parent_2:
                self.assertNotEqual(result, parent_value)
            else:
                self.assertNotEqual(result, child_value)
        
        return result