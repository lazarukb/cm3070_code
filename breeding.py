"""Holds methods to control breeding of networks and the details of that.

The main method is called to generate a crossed and mutated child network.
The sub-methods perform the actual crossing and mutation of the passed floats.

Typical usage example:

  foo = Breeding()
  bar = foo.cross_and_mutate()
"""

import population
import unittest
from copy import deepcopy
import random
import numpy as np


class Breeding(unittest.TestCase):
    """Holds methods to control breeding of networks and the details of that.
    
    The main method is called to generate a crossed and mutated child network.
    The sub-methods perform the actual crossing and mutation of the passed floats.
    """
    
    def cross_and_mutate(
        self,
        sim_population,
        serial_number,
        point_mutation_scalar,
        point_mutation_chance,
        point_mutation_amount,
        point_mutation_chance_max,
        point_mutation_amount_max,
        fitness_bias_scalar
        ):
        """Controls the cross-over and mutation to define a child Network. 

        Given the population, will select two unique Networks as parents,
        cross and mutate their weights and parameters to define a child Network,
        then return that child.

        Args:
            table_handle: An open smalltable.Table instance.
            keys: A sequence of strings representing the key of each table
            row to fetch.  String keys will be UTF-8 encoded.
            require_all_keys: If True only rows with values set for all keys will be
            returned.
            sim_population: the Population instance being evaluated.
            serial_number: current largest unique int in the population.
            point_mutation_scalar: int scaling mutation of unfit networks.
            point_mutation_chance: float probability of mutation.
            point_mutation_amount: float amount of mutation.
            point_mutation_chance_max: float maximum chance, if scaled.
            point_mutation_amount_max: float maximum amount, if scaled.
            fitness_bias_scalar: float how much to favour the more fit parent.

        Returns:
            A Network instance as defined by the dict during processing.
        """
        
        child_nn_definition = {}

        # Select two parent networks from the population, 
        #  until two unique parents are chosen
        
        parent_1 = sim_population.get_weighted_parent()
        self.assertIsInstance(parent_1, int)
        parent_2 = parent_1
        while parent_1 == parent_2:
            parent_2 = sim_population.get_weighted_parent()
            self.assertIsInstance(parent_2, int)
        self.assertNotEqual(parent_1, parent_2)
        
        # Fitness checks and modifications
        
        parent_1_fitness = sim_population.get_nn_fitness(parent_1)
        parent_2_fitness = sim_population.get_nn_fitness(parent_2)
        
        # Create the fitness bias to increase the chance of the more fit
        #  parent being selected in cross-over.  Since parent_1 is always copied
        #  to the child and parent_2 is used if the random value is less than
        #  the threshold, threshold should be higher if parent_2's fitness is.
        # So take percentage of parent_2 over the total fitness of both parents,
        #  which would be at most close to 0 or close to 1, and then subtract
        #  0.5 which leaves a result where parent_1 is more fit as a negative
        #  bias, or the opposite. Finally, scale it as required to adjust the
        #  power of the bias.
        
        fitness_bias = (parent_2_fitness / ((parent_1_fitness + parent_2_fitness)) - 0.5) * fitness_bias_scalar
        
        # If both parents have fitness of 1 then neither succeeded at the game
        # Increase the mutation chance and factor, to further distance the 
        #  child from the parent definitions.
        
        if parent_1_fitness == 1 and parent_2_fitness == 1:
            self.assertEqual(parent_1_fitness, parent_2_fitness)
            point_mutation_chance *= point_mutation_scalar
            point_mutation_amount *= point_mutation_scalar
            point_mutation_chance = min(
                point_mutation_chance,
                point_mutation_chance_max
                )
            point_mutation_amount = min(
                point_mutation_amount,
                point_mutation_amount_max
                )

        # Get the parental network definitions for future use
        
        parent_1_nn_definition = sim_population.get_neural_network_def(parent_1)
        parent_2_nn_definition = sim_population.get_neural_network_def(parent_2)
        self.assertIsInstance(parent_1_nn_definition, dict)
        self.assertIsInstance(parent_2_nn_definition, dict)
        
        # Get parental hidden layers element, assuming there is one hidden layer 
                       
        parent_1_nn_weights_bias = sim_population.get_weight_bias_definitions(parent_1, 1)
        parent_2_nn_weights_bias = sim_population.get_weight_bias_definitions(parent_2, 1)
        self.assertIsInstance(parent_1_nn_weights_bias, list)
        self.assertIsInstance(parent_2_nn_weights_bias, list)
        
        # Because the grandparents for both parents may be the same, 
        #  there is a pretty good chance that at least some of the weights
        #  in both parents are the same, inherited from the same grandparent. 
        # So we can no longer test all the weights to ensure they changed, 
        #  we have to test the checksum of the weights and activation and type 
        #  floats all added. There is almost no chance those are not unique
        #  between two networks, when measured to ~17 decimal places.
        
        self.assertNotEqual(
            parent_1_nn_definition["meta"]["checksum"],
            parent_2_nn_definition["meta"]["checksum"]
            )

        # Create a child neural network from the parents,
        #  and potentially affected by mutation.
        # Create the child hidden layer weights as a full copy of parent 1.
        # Then as necessary, replace with values from parent 2,
        #  thus achieving crossover breeding.
        
        child_network_weights_bias = deepcopy(parent_1_nn_weights_bias)
        child_network_weights = child_network_weights_bias[0]                             ##### currently assuming that there is only one hidden layer

        # This assert checks that the parents are fully different 
        #  because their weight checksums are different.
        
        self.assertNotEqual(
            parent_1_nn_definition["meta"]["checksum"],
            parent_2_nn_definition["meta"]["checksum"]
            )
        
        # Iterate through child hidden layer weights,
        #  applying crossover and mutation to each
        
        completed_iterations = 0
        
        # pass child_network_weights[0] and parent_2_nn_weights_bias[0][0],
        #  and weighted chance of parent_2 being selected
        
        for i in range (len(child_network_weights[0])):
            # Crossover
            
            child_network_weights[0][i] = self.c_and_m(
                "weight",
                child_network_weights[0][i],
                parent_2_nn_weights_bias[0][0][i],
                point_mutation_chance,
                point_mutation_amount,
                fitness_bias
                )
            completed_iterations += 1
            
            # Make sure the neuron weights are within the acceptable bounds,
            #  whether or not they were mutated.
            
            self.assertIsInstance(child_network_weights[0][i], np.float32)
            self.assertGreaterEqual(child_network_weights[0][i], -1.0)
            self.assertLessEqual(child_network_weights[0][i], 1.0)
        
        # Make sure we went over all the neurons
        
        self.assertEqual(completed_iterations, len(child_network_weights[0]))
        
        # Write the new weights back to the weights+biases variable, 
        #  for later writing to the network object.
        
        child_network_weights_bias[0] = child_network_weights
        
        # Now crossover and mutate the floats in the DNA
        # Again start with the child being a full copy of parent 1,
        #  then achieve crossover by replacing with parts of parent_2.
        
        child_nn_definition = deepcopy(parent_1_nn_definition)
        
        # Nothing in the input layer to crossover or mutate.
        #  If we change the inputs then the network will not be able to
        #  interact with OpenAI gym and its fitness will be 0, so it will "die".
        #  No point in doing that.
        
        # Iterate through the child hidden layer floats in the DNA,
        #  applying crossover and mutation.
        
        for layer in range(len(child_nn_definition["hidden_layers"])):
            # Crossover, for each type and activation
            
            child_nn_definition["hidden_layers"][layer]["activation"] = self.c_and_m(
                "definition",
                child_nn_definition["hidden_layers"][layer]["activation"],
                parent_2_nn_definition["hidden_layers"][layer]["activation"],
                point_mutation_chance,
                point_mutation_amount,
                fitness_bias
                )
            child_nn_definition["hidden_layers"][layer]["type"] = self.c_and_m(
                "definition",
                child_nn_definition["hidden_layers"][layer]["type"],
                parent_2_nn_definition["hidden_layers"][layer]["type"],
                point_mutation_chance,
                point_mutation_amount,
                fitness_bias
                )
            
            # Validate that all the floats are within the acceptable ranges.
            
            self.assertIsInstance(
                child_nn_definition["hidden_layers"][layer]["type"],
                float
                )
            self.assertIsInstance(
                child_nn_definition["hidden_layers"][layer]["activation"],
                float
                )
            self.assertGreaterEqual(
                child_nn_definition["hidden_layers"][layer]["type"],
                0.0
                )
            self.assertLessEqual(
                child_nn_definition["hidden_layers"][layer]["type"],
                1.0
                )
            self.assertGreaterEqual(
                child_nn_definition["hidden_layers"][layer]["activation"],
                0.0
                )
            self.assertLessEqual(
                child_nn_definition["hidden_layers"][layer]["activation"],
                1.0
                )
            
        # Crossover, for each type and activation     
           
        child_nn_definition["output"]["type"] = self.c_and_m(
            "definition",
            child_nn_definition["output"]["type"],
            parent_2_nn_definition["output"]["type"],
            point_mutation_chance,
            point_mutation_amount,
            fitness_bias
            )       
        child_nn_definition["output"]["activation"] = self.c_and_m(
            "definition",
            child_nn_definition["output"]["activation"],
            parent_2_nn_definition["output"]["activation"],
            point_mutation_chance,
            point_mutation_amount,
            fitness_bias
            )
            
               
        # Validate that all the floats are within the acceptable ranges.
        
        self.assertIsInstance(child_nn_definition["output"]["type"], float)
        self.assertIsInstance(child_nn_definition["output"]["activation"], float)
        self.assertGreaterEqual(child_nn_definition["output"]["type"], 0.0)
        self.assertLessEqual(child_nn_definition["output"]["type"], 1.0)
        self.assertGreaterEqual(child_nn_definition["output"]["activation"], 0.0)
        self.assertLessEqual(child_nn_definition["output"]["activation"], 1.0)   

        # This will retrieve the last elements of the layers information,
        #  which will be the output layer.
        # Continues to assume there is only one hidden layer
        #  which needs to be changed later on.
        # This layer also has 512 weights #####################################################################################
        
        parent_1_nn_output_weights_bias = sim_population.get_weight_bias_definitions(parent_1, 2)
        parent_2_nn_output_weights_bias = sim_population.get_weight_bias_definitions(parent_2, 2)
                
        # Create the child output layer weights as a full copy of parent 1.
        # Then if necessary, replace with values from parent 2,
        #  thus achieving crossover breeding.
        
        child_network_output_weights_bias = deepcopy(parent_1_nn_output_weights_bias)
        child_network_output_weights = child_network_output_weights_bias[0]       ##### currently assuming that there is only one hidden layer
        child_network_output_weights_initial = deepcopy(child_network_output_weights)
        
        # Randomly sample to make sure the copy worked
        
        for i in range(10):
            rand = random.randint(
                0,
                len(parent_1_nn_output_weights_bias[0][0]) - 1
                )
            self.assertEqual(
                child_network_output_weights[0][rand],
                parent_1_nn_output_weights_bias[0][0][rand]
                )
        
        # Iterate through the child output layer weights,
        #  applying crossover and mutation to each
        
        completed_iterations = 0
        for i in range (len(child_network_output_weights[0])):
            # Crossover
            
            child_network_output_weights[0][i] = self.c_and_m(
                "weight",
                child_network_output_weights[0][i],
                parent_2_nn_output_weights_bias[0][0][i],
                point_mutation_chance,
                point_mutation_amount,
                fitness_bias
                )
            completed_iterations += 1
        
            # Make sure the neuron weights are within the acceptable bounds,
            #  whether or not they were mutated
            
            self.assertGreaterEqual(child_network_output_weights[0][i], -1.0)
            self.assertLessEqual(child_network_output_weights[0][i], 1.0)
            
        # Make sure we went over all the neurons
        
        self.assertEqual(completed_iterations, len(child_network_output_weights[0]))
        
        # Write the new weights back to the weights+biases variable,
        #  for later writing to the network object.       
             
        child_network_output_weights_bias[0] = child_network_output_weights
        
        # Get parent serial numbers and write everything to the child definition
        
        parent_1_sn = parent_1_nn_definition['meta']['serial_number']
        parent_2_sn = parent_2_nn_definition['meta']['serial_number']
        child_nn_obj = sim_population.create_nn(
            serial_number,
            child_nn_definition,
            child_network_weights_bias,
            child_network_output_weights_bias,
            parent_1_sn, parent_2_sn
            )
        
        # Assertions to confirm the child nn is properly defined and populated
        
        self.assertIsInstance(child_nn_obj.get_network_dna()["inputs"], int)
        self.assertIsNot(len(child_nn_obj.get_network_dna()["hidden_layers"]), 0)
        self.assertIsInstance(child_nn_obj.get_network_dna()["output"], dict)
        self.assertIsNot(child_nn_obj.checksum_weights(1), 0)
        self.assertIsNot(child_nn_obj.checksum_weights(2), 0)
    
        return child_nn_obj
    
    
    def carry_over():
        pass
    
    
    def c_and_m(
        self,
        weight_or_def,
        child_value,
        parent_value,
        point_mutation_chance,
        point_mutation_amount,
        fitness_bias=0
        ):
        """Crosses and mutates the two float values provided.
        
        remove this: A docstring should give enough information to write a call to the function without reading the function’s code. The docstring should describe the function’s calling syntax and its semantics, but generally not its implementation details, unless those details are relevant to how the function is to be used

        Args
            weight_or_def: string separating if weight or definition is the input.
            child_value: current float of the child network, copied from parent_1.
            parent_value: parent_2 float value to be considered.
            point_mutation_chance: float chance of mutation, previously scaled.
            point_mutation_amount: float maximum amount of mutation, previously scaled.
            fitness_bias: float bias to the more fit of the parents.

        Returns:
            Float, or numpy.float depending on the input, representing the value
            to be used for the child network weight or definition.
        """
        
        # Weights have a range from -1.0 to 1.0
        # Definitions have a range from 0.0 to 1.0
        # Make sure the type was sent properly
        
        self.assertIn(weight_or_def, ["weight", "definition"])
        if weight_or_def == "weight":
            range_min = -1.0
            range_max = 1.0
        elif weight_or_def == "definition":
            range_min = 0.0
            range_max = 1.0
        
        # Choose either the child value, which is currently a copy of parent_1,
        #  or the parent_2 value
        
        rand = random.random()
        used_parent_2 = None
        if fitness_bias <= 0:
            threshold = max(0.5 + fitness_bias, -1.0)
        else:
            threshold = min(0.5 + fitness_bias, 1.0)
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
        if mutated and result != range_min and result != range_max:
            if used_parent_2:
                self.assertNotEqual(result, parent_value)
            else:
                self.assertNotEqual(result, child_value)
        
        return result