import genome
import keras
from copy import deepcopy
from keras import layers

class Network:
    def __init__(self):
        self.genome = genome.Genome.get_gene_specifications()
        self.fitness = None
        self.weights = []
        self.dna = {"meta": {"serial_number": 0, "checksum": 0, "parent_1": None, "parent_2": None, "hidden_weights": None, "output_weights": None}, "inputs": None, "hidden_layers": [], "output": None}
    
    '''Create a random network from the genome'''
    def create_random_network_dna(self, serial_number, inputs_size, hidden_layers_count = 1):
        self.dna = genome.Genome.create_random_genome(self.genome, hidden_layers_count)
        self.dna["meta"]["serial_number"] = serial_number
        self.dna["inputs"] = inputs_size
        self.weights = []
    
    '''Return the components of this network in addressable DICT format'''
    def get_network_dna(self):
        return self.dna
    
    '''Set the components of this network via DICT inputs'''
    def create_specified_network_dna(self, serial_number, input_settings, hidden_weights, output_weights, parent_1, parent_2):
        self.dna["inputs"] = input_settings["inputs"]
        self.dna["hidden_layers"] = input_settings["hidden_layers"]
        self.dna["output"] = input_settings["output"]
        self.dna["meta"]["serial_number"] = serial_number
        self.save_weight_bias_definitions(1, hidden_weights)
        self.save_weight_bias_definitions(2, output_weights)
        self.dna["meta"]["hidden_checksum"] = self.checksum_weights(1)
        self.dna["meta"]["output_checksum"] = self.checksum_weights(2)
        self.dna["meta"]["checksum"] = self.checksum()
        self.dna["meta"]["parent_1"] = parent_1
        self.dna["meta"]["parent_2"] = parent_2
    
    '''Return the fitness score for this network'''
    def get_fitness(self):
        return self.fitness
    
    '''Set the fitness score for this network'''
    def set_fitness(self, _fitness):
        self.fitness = _fitness
        
    '''Get the weights of a specified layer from this network'''
    def get_weight_bias_definitions(self, layer):
        # return the requested layer - 1, because layer 0 is the inputs and has no weights
        return self.weights[layer - 1]
    
    '''Save the weights of a specified layer back to this network'''
    def save_weight_bias_definitions(self, layer, _weights):
        if len(self.weights) == 0 or len(self.weights) < layer + 1:
            self.weights.append(_weights)
        else:
            self.weights[layer] = _weights
    
    '''Builds the keras nn.Model from self and returns it'''
    def get_network_model(self):
        # Define the input layer from the dna
        inputs = layers.Input(shape = (self.dna["inputs"],))
        
        # Get the definitions for the hidden layers
        hidden_layer_definitions = self.dna["hidden_layers"]
        
        # Create the hidden layers
        hidden_layers = []

        # Link the first hidden layer to the input layer
        # determine the keyword for the first hidden layer activation type
        activation_type = self.get_activation_function_keyword(hidden_layer_definitions[0]["activation"])
        
        # Create and link the new hidden layer to the input layer
        if hidden_layer_definitions[0]["type"] >= 0.0 and hidden_layer_definitions[0]["type"] <= 1.0:
            new_layer = layers.Dense(hidden_layer_definitions[0]["neurons"], activation_type)(inputs)
            hidden_layers.append(new_layer)
            
        # If they exist, link additional hidden layers in sequence
        if len(hidden_layer_definitions) > 1:
            for hidden_layer in range(1, len(hidden_layer_definitions)):
                activation_type = self.get_activation_function_keyword(hidden_layer_definitions[hidden_layer]["activation"])
                if hidden_layer_definitions[hidden_layer]["type"] >= 0.0 and hidden_layer_definitions[hidden_layer]["type"] <= 1.0:
                    new_layer = layers.Dense(hidden_layer_definitions[hidden_layer]["neurons"], activation_type)(hidden_layers[hidden_layer - 1])
                    hidden_layers.append(new_layer)
                    
                    
        # Define the output layer from the dna
        activation_type = self.get_activation_function_keyword(self.dna["output"]["activation"])

        # Create and link the output layer to the last hidden layer
        if self.dna["output"]["type"] >= 0.0 and self.dna["output"]["type"] <= 1.0:
            outputs = layers.Dense(self.dna["output"]["count"], activation=activation_type)(hidden_layers[len(hidden_layers) - 1])

        # Build the model
        nn_model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Right now the weights have been randomly generated.
        # If there are already weights in this instance then overwrite with those
        # If there are no weights then this is a new instance - recover and save the randomised weights
        if len(self.weights) != 0:
            pass
        else:
            # Gather the weights and biases of the layers, input, all hidden, and output layers are included in the method. The first is the input, which has no weights, so skip it.
            for layer in range(1, len(nn_model.layers) - 0):
                weights_biases = (nn_model.layers[layer].get_weights())
                layer_config = (nn_model.layers[layer].get_config())
                self.save_weight_bias_definitions(layer, weights_biases)
                
        # Update the metadata for this instance with the checksum weights of the layers.      
        self.dna["meta"]["hidden_checksum"] = self.checksum_weights(1)
        self.dna["meta"]["output_checksum"] = self.checksum_weights(2)
        self.dna["meta"]["checksum"] = self.checksum()
        
        return nn_model
        
        
    '''Generates and returns a checksum of the weights of the noted layer, for validation that they exist and are not all the same'''   
    def checksum_weights(self, layer):
        weights_biases = self.get_weight_bias_definitions(layer)
        sumOfList = sum(weights_biases[0][0])
        return sumOfList
    
    
    '''Generates and returns a checksum of many different numbers in the network'''
    def checksum(self):
        total = 0
        total += self.checksum_weights(1)
        total += self.checksum_weights(2)
        total += self.dna["hidden_layers"][0]["type"]
        total += self.dna["hidden_layers"][0]["activation"]
        total += self.dna["output"]["type"]
        total += self.dna["output"]["activation"]
        return total
             
        
    def get_activation_function_keyword(self, _activation):
        ''' Takes input of the float value of the activation gene, returns the keyword relating to that activation function'''
        match _activation:
            case _activation if 0.0 <= _activation < 0.25:
                activation_type = "relu"
            case _activation if 0.25 <= _activation < 0.50:
                activation_type = "linear"
            case _activation if 0.50 <= _activation <= 0.75:
                activation_type = "sigmoid"
            case _activation if 0.75 <= _activation <= 1.0:
                activation_type = "tanh"
        return activation_type
    