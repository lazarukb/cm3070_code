'''
Code for creating and modifying the genome that is used to define a neural network
Fixed components of the genome:
Inputs: this is the number of 
'''

# import numpy as np
import random

# this code heavily derived from code presented in lectures in CM3020
class Genome():
    def __init__(self):
        pass
        # self.gene_specification = {
        #     "inputs": 1,
        #     "hidden_type": 1,
        #     "hidden_neurons_min": 512,
        #     "hidden_neurons_max": 512,
        #     "hidden_activation": 1,
        #     "outputs": 3,
        #     "output_type": 1,
        #     "output_activation": 1            
        # }
    
    # This one is properly static as this is effectively a data class, never itself instantiated.
    @staticmethod
    def create_random_genome(specification, hidden_layers_count):
        genome = {}
        
        # Prepare for storing some metadata
        genome["meta"] = {"serial_number": None, "checksum": None, "parent_1": None, "parent_2": None, "hidden_weights": None, "output_weights": None}
        
        # Define the hidden layers
        hidden_layer_definitions = []
        for layer in range(hidden_layers_count):
            activation = random.uniform(0, specification["hidden_activation"])
            type = random.uniform(0, specification["hidden_type"])
            neurons = random.uniform(specification["hidden_neurons_min"], specification["hidden_neurons_max"])
            layer = {"type": type, "neurons": neurons, "activation": activation}
            hidden_layer_definitions.append(layer)
        genome["hidden_layers"] = hidden_layer_definitions
        
        # Final components of the genome
        output_count = (specification["outputs"])
        activation_out = random.uniform(0, specification["output_activation"])
        type_out = random.uniform(0, specification["output_type"])
        output_layer = {"type": type_out, "count": output_count, "activation": activation_out}
        genome["output"] = output_layer
        
        return genome
    

    '''Defines the components of the input, hidden, and output layers, along with their maximum values'''
    # This method heavily inspired from CM3020 lectures
    @staticmethod
    # def get_gene_specifications(self):
    def get_gene_specifications():
        gene_specification = {
            # "inputs": 40,
            "hidden_type": 1,
            "hidden_neurons_min": 512,
            "hidden_neurons_max": 512,
            "hidden_activation": 1,
            "outputs": 5,
            "output_type": 1,
            "output_activation": 1
        }
        return gene_specification
    