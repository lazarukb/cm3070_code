"""Contains the Network objects and allows access to them.
"""

import network
import random

class Population():
    """Contains the Network objects and allows access to them.

    Contains the data of the neural network objects.
    Methods for getting and setting actions with those Networks.

    Attributes:
    remove this: only need to be here if there is an __init__
        fitness_map: a list of the weighted fitnesses of this population.
        networks: a list of Network instances.
    """
    
    def __init__(self):
        self.fitness_map = []
        self.networks = []
    
    def create_random_population(self, population_size, serial_number, inputs_size):
        """Generate a population of Network objects with randomised DNA.

        Args:
            population_size: INT count of the number of Networks to instantiate.
            serial_number: INT unique identifier to be assigned to the Network.
            inputs_size: INT value of the input tensor size to be assigned.

        Returns:
            None. Instantiates the Network and stores it in self.networks[].
        """
        
        for i in range(population_size):
            temp_network = network.Network()
            temp_network.create_random_network_dna(serial_number + i, inputs_size)
            self.networks.append(temp_network)
    
    def get_population_size(self):
        """Getter to return the number of neural networks in the population.

        Args:
            None

        Returns:
            INT of the number of neural network objects stored in this Population.
        """
        
        return len(self.networks)
    
    def get_neural_network_def(self, network_id):
        """Getter to return a neural network definition from the population.

        Args:
            network_id: INT of the element in the self.networks list to return.

        Returns:
            The DICT of the DNA used to define the specified Network model.
        """
        
        return (self.networks[network_id].get_network_dna())
    
    def set_nn_fitness(self, network_id, fitness):
        """Setter to store the fitness of a Network in the Population.

        Args:
            network_id: INT of the element in the self.networks list to update.
            fitness: INT of the fitness to be assigned.

        Returns:
            None. Modifies the Network definition.
        """
        
        self.networks[network_id].set_fitness(fitness)
        
    def get_nn_fitness(self, network_id):
        """Getter to return a neural network fitness from the Population.

        Args:
            network_id: INT of the element in the self.networks list to query.

        Returns:
            The INT of the fitness stored with that Network instance.
        """
        
        return self.networks[network_id].get_fitness()
    
    def create_fitness_map(self):
        """Create/replace the fitness map for this Population.

        Args:
            None.

        Returns:
            None. Updates the fitness map stored in self.
        """
        
        total_fitness = 0
        if len(self.fitness_map) > 0:
            self.fitness_map = []
        for i in range(self.get_population_size()):
            total_fitness += self.get_nn_fitness(i)
            self.fitness_map.append(total_fitness)
    
    def get_fitness_map(self):
        """Getter to return the fitness map for this Population.

        Args:
            None.

        Returns:
            The list of the fitness map stored in self.
        """
        
        return self.fitness_map
    
    def get_weighted_parent(self):
        """Selects and returns one network from the weighted fitness map.

        Randomly chooses an element from the fitness map, which is a weighted
         list of fitnesses for each Network in this Population.
        This code and concept of the fitness map heavily derived from
         CM3020 lectures and provided code.
         
        Args:
            None.

        Returns:
            An INT of one Network from the fitness map stored in self.
        """
        
        # Create the seed and weight it against the last value in the
        #  fitness map which will be the sum of all fitnesses
        
        random_seed = random.random() * self.fitness_map[-1]
        for network in range(len(self.fitness_map)):
            if random_seed <= self.fitness_map[network]:
                return network
        
    def save_weight_bias_definitions(self, network_id, layer, _weights):
        """Set the weight definitions of a given layer.

        Given the provided weights, save them to the Network instance definition.

        Args:
            network_id: INT of the element of self.networks to be modified.
            layer: layer INT to be updated.
            _weights: numpy array of the weights to be stored.

        Returns:
            None. Updates the Network instance stored in self.
        """
        
        self.networks[network_id].save_weight_bias_definitions(layer, _weights)
        
    '''Get the weights definitions of a given layer on a given network in this population'''
    def get_weight_bias_definitions(self, network_id, layer):
        """Getter to return the weights of a specified layer.

        Args:
            network_id: INT of the element of self.networks to be queried.
            layer: layer INT to be queried.

        Returns:
            A list of the stored weights and biases.
        """
        
        return self.networks[network_id].get_weight_bias_definitions(layer)
    
    '''Get the specified nn model object'''
    def get_neural_network_model(self, network_id):
        """Builds and returns a Keras neural network model.

        Triggers the Network instance to read the various definition parameters
         stored in itself and instantiate an Keras neural network model from that,
        and return it.

        Args:
            network_id: INT of the element of self.networks to be queried.

        Returns:
            A Keras neural network model instance.
        """
        
        return self.networks[network_id].get_network_model()
    
    '''Create a nn from provided specs'''
    def create_nn(self, serial_number, specs, hidden_weights, output_weights, parent_1, parent_2):
        """Instantiates a new Network object and returns it.

        Args:
            serial_number: INT of the serial number to be assigned to this Network.
            specs: DICT specifications of the DNA to be stored.
            hidden_weights: list of weights in the hidden layers.
            output_weights: list of weights in the output layers.
            parent_1: INT of the serial number from the first parent Network.
            parent_2: INT of the serial number from the second parent Network.

        Returns:
            A Network instance as defined by the dict during processing.
        """
        
        temp_network = network.Network()
        temp_network.create_specified_network_dna(serial_number, specs, hidden_weights, output_weights, parent_1, parent_2)
        return temp_network
    
    '''Add a nn to this population'''
    def add_nn(self, network):
        """Setter to store a neural network object in self.networks.

        Args:
            network: Network instance.

        Returns:
            None. Updates the self.networks.
        """
        
        self.networks.append(network)
    