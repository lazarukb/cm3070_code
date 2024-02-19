"""Defines and controls a neural network instance, using Keras.
"""

import genome
import keras
from keras import layers

class Network:
    """Defines and controls a neural network instance, using Keras.

    Attributes:
        genome: dicts of the fundamental parts of the network definition.
        fitness: int of the fitness score achieved by a Network instance.
        weights: list of numpy arrays for layer weights and biases.
        dna: dict of properties specific to this instance of the Network.
    """
    
    def __init__(self):
        self.genome = genome.Genome.get_gene_specifications()
        self.fitness = None
        self.weights = []
        self.dna = {
            "meta":{
                "serial_number": 0,
                "checksum": 0,
                "parent_1": None,
                "parent_2": None,
                "hidden_weights": None,
                "output_weights": None
                },
            "inputs": None,
            "hidden_layers": [],
            "output": None
            }
    
    def create_random_network_dna(
        self,
        serial_number,
        inputs_size,
        hidden_layers_count = 1
        ):
        """Builds random DNA values and stores in the instance.

        Args:
            serial_number: unique int to be assigned to this instance.
            inputs_size: int of size of input tensor the network should expect.
            hidden_layers_count: int of the number of hidden layers to create.

        Returns:
            None
        """
        
        self.dna = genome.Genome.create_random_genome(
            self.genome,
            hidden_layers_count
            )
        self.dna["meta"]["serial_number"] = serial_number
        self.dna["inputs"] = inputs_size
        self.weights = []
    
    def get_network_dna(self):
        """Getter to return the instance DNA.

        Args:
            None

        Returns:
            Dict of the DNA of this instance.
        """
        
        return self.dna
    
    def create_specified_network_dna(
        self,
        serial_number,
        input_settings,
        hidden_weights,
        output_weights,
        parent_1,
        parent_2
        ):
        """Save the components of this instance to the DNA attribute.

        Args:
            serial_number: unique INT to be assigned to this instance.
            input_settings: DICT of input parameters
            hidden_weights: numpy array, or blank if not initialised.
            output_weights: numpy array, or blank if not initialised.
            parent_1: INT serial number of the parent_1 Network.
            parent_2: INT serial number of the parent_2 Network.

        Returns:
            None. Modifies the instance.
            Also creates and stores the instance checksum value.
        """
        
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
    
    def get_fitness(self):
        """Getter to return the stored fitness score for this instance.

        Args:
            None.

        Returns:
            INT (natural number) of the fitness stored in this instance
        """
        
        return self.fitness
    
    def set_fitness(self, _fitness):
        """Setter to save the stored fitness score for this instance.

        Args:
            _fitness: INT (natural number) to be saved as the fitness.

        Returns:
            None. Modifies the instance.
        """
        
        self.fitness = _fitness
        
    def get_weight_bias_definitions(self, layer):
        """Get the weights of a specified layer from this instance.

        Ignores the first stored layer as that is the input layer, 
         which has no weights and so is not stored here. Layer 1 is the hidden
         layer, and layer 2 is the output layer. In the attribute these are
         [0] and [1].
         
        Args:
            layer: INT (whole number) specifying the layer to be returned.

        Returns:
            A numpy array of the weights and biases.
        """
        
        return self.weights[layer - 1]
    
    def save_weight_bias_definitions(self, layer, _weights):
        """Save the weights of a specified layer back to this instance.

        Appends the weights as necessary or if the layer is already defined,
         overwrites the existing information with the new array.

        Args:
            layer: INT (whole number) of the layer to be updated, or created.
            _weights: array of the weights to be stored.

        Returns:
            None. Modifies the instance.
        """
        
        if len(self.weights) == 0 or len(self.weights) < layer + 1:
            self.weights.append(_weights)
        else:
            self.weights[layer] = _weights
    
    def get_network_model(self):
        """Builds the keras nn.Model of this instance, updating checksums.
        
        As the model is initialised the weights and biases are randomly seeded.
        If the instance has previously stored weights and biases those are used
         to overwrite the random ones. If not, the random ones are saved to the
         instance for future use.

        Args:
            None.

        Returns:
            A Keras nn.Model object of the instance. Also recalculates and saves
             the instance checksum in the case that this model is being called
             for the first time, as that is when the layer weights and biases
             are initialised.
        """
        
        # Get parameters from the instance DNA and initialise.
        
        inputs = layers.Input(shape = (self.dna["inputs"],))
        hidden_layer_definitions = self.dna["hidden_layers"]
        hidden_layers = []

        # Determine the first hidden layer activation type
        
        activation_type = self.get_activation_function_keyword(
            hidden_layer_definitions[0]["activation"]
            )
        
        # Create and link the new hidden layer to the input layer
        
        if hidden_layer_definitions[0]["type"] >= 0.0 and hidden_layer_definitions[0]["type"] <= 1.0:
            new_layer = layers.Dense(
                hidden_layer_definitions[0]["neurons"],
                activation_type
                )(inputs)
            hidden_layers.append(new_layer)
            
        # If they exist, link additional hidden layers in sequence
        
        if len(hidden_layer_definitions) > 1:
            for hidden_layer in range(1, len(hidden_layer_definitions)):
                activation_type = self.get_activation_function_keyword(
                    hidden_layer_definitions[hidden_layer]["activation"]
                    )
                if hidden_layer_definitions[hidden_layer]["type"] >= 0.0 and hidden_layer_definitions[hidden_layer]["type"] <= 1.0:
                    new_layer = layers.Dense(
                        hidden_layer_definitions[hidden_layer]["neurons"],
                        activation_type
                        )(hidden_layers[hidden_layer - 1])
                    hidden_layers.append(new_layer)
                    
        # Define the output layer from the dna
        
        activation_type = self.get_activation_function_keyword(self.dna["output"]["activation"])

        # Create and link the output layer to the last hidden layer
        
        if self.dna["output"]["type"] >= 0.0 and self.dna["output"]["type"] <= 1.0:
            outputs = layers.Dense(
                self.dna["output"]["count"], activation=activation_type
                )(hidden_layers[len(hidden_layers) - 1])

        # Build the model
        
        nn_model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Right now the weights have been randomly generated.
        # If there are already weights in this instance then overwrite.
        # If there are no weights then this is a new instance - 
        #  recover and save the randomised weights
        
        if len(self.weights) != 0:
            pass
        else:
            # Gather the weights and biases of the layers, input, all hidden, 
            #  and output layers are included in the method. The first is the 
            #  input, which has no weights, so skip it.
            
            for layer in range(1, len(nn_model.layers) - 0):
                weights_biases = (nn_model.layers[layer].get_weights())
                layer_config = (nn_model.layers[layer].get_config())
                self.save_weight_bias_definitions(layer, weights_biases)
                
        # Update  metadata for this instance with  checksum weights of layers. 
             
        self.dna["meta"]["hidden_checksum"] = self.checksum_weights(1)
        self.dna["meta"]["output_checksum"] = self.checksum_weights(2)
        self.dna["meta"]["checksum"] = self.checksum()
        
        return nn_model
           
    def checksum_weights(self, layer):
        """Generates and returns a checksum of the weights of the noted layer.

        In the Keras array, the data is in element [0], and the 
         the weights are stored the data element 0, biases in element 1.
        Ignoring the biases here.

        Args:
            layer: the INT layer number to be queried, expected to be a whole number.

        Returns:
            The float sum of all the weights in the queried nn layer.
        """
        
        weights_biases = self.get_weight_bias_definitions(layer)
        sumOfList = sum(weights_biases[0][0])
        return sumOfList
    
    
    '''Generates and returns a checksum of many different numbers in the network'''
    def checksum(self):
        """Generates the checksum of the instance.

        Gathers values from layer checksums and definition parameters and 
         sums them to create a unique checksum for this instance.
         
        Args:
            None.

        Returns:
            A float of the checksum for this instance.
        """
        
        total = 0
        total += self.checksum_weights(1)
        total += self.checksum_weights(2)
        total += self.dna["hidden_layers"][0]["type"]
        total += self.dna["hidden_layers"][0]["activation"]
        total += self.dna["output"]["type"]
        total += self.dna["output"]["activation"]
        return total
             
        
    def get_activation_function_keyword(self, _activation):
        """Determines keyword matching the activation function definition value.
        
        Args:
            _activation: float value between 0.0 - 1.0

        Returns:
            String keyword of the activation function type.
        """
        
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
    