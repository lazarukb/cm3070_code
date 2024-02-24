"""Defines the foundational structure of the neural network.

This is a mixed data and setter to store the foundational definition of a neural
 network along with the maximum values that those components can have. Also a 
 method here to populate a nn definition from those limits.
"""

import random

# this code heavily derived from code presented in lectures in CM3020
class Genome():
    """Stores the foundation for defining a network, and randomly initialises.

    Will return a DNA for an instance derived from the bounds of the genome.
    """
    
    @staticmethod
    def create_random_genome(specification, hidden_layers_count):
        """Creates a random DNA from the limitations of the genome.

        Args:
            specification: DICT of the DNA, defining the bounds of each key.
            hidden_layers_count: INT count of hidden layers to be added.

        Returns:
            A DICT with DNA values for a network instance.
        """
        
        # Prepare for storing some metadata
        genome = {}
        genome["meta"] = {
            "serial_number": None,
            "checksum": None,
            "parent_1": None,
            "parent_2": None,
            }
        
        # Define the hidden layers
        hidden_layer_definitions = []
        for layer in range(hidden_layers_count):
            activation = random.uniform(0, specification["hidden_activation"])
            type = random.uniform(0, specification["hidden_type"])
            neurons = random.uniform(
                specification["hidden_neurons_min"],
                specification["hidden_neurons_max"]
                )
            layer = {"type": type, "neurons": neurons, "activation": activation}
            hidden_layer_definitions.append(layer)
        genome["hidden_layers"] = hidden_layer_definitions
        
        # Final components of the genome
        output_count = (specification["outputs"])
        activation_out = random.uniform(0, specification["output_activation"])
        type_out = random.uniform(0, specification["output_type"])
        output_layer = {
            "type": type_out,
            "count": output_count,
            "activation": activation_out
            }
        genome["output"] = output_layer
        
        return genome
    

    @staticmethod
    def get_gene_specifications():
        """Data, defining the components of a nn instance and their maximum values.
        
        This method structure heavily inspired from CM3020 lectures.
        
        Args:
            None

        Returns:
            DICT of the nn components and their maximum values.
        """
        
        gene_specification = {
            "hidden_type": 1,
            "hidden_neurons_min": 512,
            "hidden_neurons_max": 512,
            "hidden_activation": 1,
            "outputs": 5,
            "output_type": 1,
            "output_activation": 1
        }
        
        return gene_specification
    