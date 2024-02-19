# import unittest
import network
import random

class Population():
    def __init__(self):
        self.fitness_map = []
        self.networks = []
    
    '''Generate a population of Network objects with randomised dna'''
    def create_random_population(self, population_size, serial_number, inputs_size):
        for i in range(population_size):
            temp_network = network.Network()
            temp_network.create_random_network_dna(serial_number + i, inputs_size)
            self.networks.append(temp_network)
    
    '''Returns the number of neural networks in the population'''
    def get_population_size(self):
        return len(self.networks)
    
    '''Return the specified neural network from the population'''
    def get_neural_network_def(self, network_id):
        return (self.networks[network_id].get_network_dna())
    
    '''Set the fitness of a network in the population'''
    def set_nn_fitness(self, network_id, fitness):
        self.networks[network_id].set_fitness(fitness)
        
    '''Get the fitness of a network in the population'''
    def get_nn_fitness(self, network_id):
        return self.networks[network_id].get_fitness()
    
    '''Create/replace the fitness map for this population'''
    def create_fitness_map(self):
        total_fitness = 0
        # Reset if necessary
        if len(self.fitness_map) > 0:
            self.fitness_map = []
        for i in range(self.get_population_size()):
            total_fitness += self.get_nn_fitness(i)
            self.fitness_map.append(total_fitness)
        # print(f"Finished, this population's network map is {self.fitness_map}")
    
    '''Return the fitness map for this population'''
    def get_fitness_map(self):
        return self.fitness_map
    
    '''Select a parent neural network from the fitness map
    returns integer, the number of the network in the population'''
    def get_weighted_parent(self):
        # Create the seed and weight it against the last value in the
        #  fitness map which will be the sum of all fitnesses
        random_seed = random.random() * self.fitness_map[-1]
        for network in range(len(self.fitness_map)):
            if random_seed <= self.fitness_map[network]:
                return network
        
    '''Set the weight definitions of a given layer on a given network in this population'''
    def save_weight_bias_definitions(self, network_id, layer, _weights):
        self.networks[network_id].save_weight_bias_definitions(layer, _weights)
        
    '''Get the weights definitions of a given layer on a given network in this population'''
    def get_weight_bias_definitions(self, network_id, layer):
        return self.networks[network_id].get_weight_bias_definitions(layer)
    
    '''Get the specified nn model object'''
    def get_neural_network_model(self, network_id):
        return self.networks[network_id].get_network_model()
    
    '''Create a nn from provided specs'''
    def create_nn(self, serial_number, specs, hidden_weights, output_weights, parent_1, parent_2):
        temp_network = network.Network()
        temp_network.create_specified_network_dna(serial_number, specs, hidden_weights, output_weights, parent_1, parent_2)
        return temp_network
    
    '''Add a nn to this population'''
    def add_nn(self, network):
        self.networks.append(network)
    