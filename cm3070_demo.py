import keras
import numpy as np
import random

from keras import layers

def get_activation_function_keyword(_activation):
    ''' Takes input of the float value of the activation gene, returns the keyword relating to that activation function
    '''
    match _activation:
        case _activation if 0.0 <= _activation < 0.33:
            activation_type = "relu"
        case _activation if 0.33 <= _activation < 0.67:
            activation_type = "linear"
        case _activation if 0.67 <= _activation <= 1.0:
            activation_type = "sigmoid"
    return activation_type


# define genes through floats so they can be evolved or mutated
# the layer gene:
    # type: [0-1] = dense
    # neurons: integer
    # activation_function: [0-0.33] = relu, 0.33-0.67 = linear, 0.67-1.0 = sigmoid

# Definitions for the input layer gene
shape_input = (1,)


# Definitions for the hidden layer genes - current set to always return one hidden layer
hidden_layers_count = random.randint(1, 1)
hidden_layer_definitions = []

print(hidden_layers_count)

for layer in range(hidden_layers_count):
    activation = random.random()
    type = random.random()
    layer = {"type": type, "neurons": 512, "activation": activation}
    print("\n\nHidden layer:")
    print(layer)
    print("\n")
    hidden_layer_definitions.append(layer)
    
print(len(hidden_layer_definitions))
    

# activation_1 = random.random()
# type_1 = random.random()
# layer_gene_1 = {"type": type_1, "neurons": 512, "activation": activation_1}


# Definitions for the output layer gene
activation_out = random.random()
type_out = random.random()
layer_output = {"type": type_out, "count": 3, "activation": activation_out}
print("\n\nOutput layer:")
print(layer_output)
print("\n")


# Final genome
genome = {"input": shape_input, "hidden_layers": hidden_layer_definitions, "output": layer_output}



# Define the individual neural network based on the genome

inputs = layers.Input(shape=genome["input"])
# inputs = layers.Input(shape=(1,))

# Create the hidden layers
hidden_layers = []

print(hidden_layer_definitions[0]["neurons"])

# Link the first hidden layer to the input layer
# determine the keyword for the activation type
activation_type = get_activation_function_keyword(hidden_layer_definitions[0]["activation"])
if hidden_layer_definitions[0]["type"] >= 0.0 and hidden_layer_definitions[0]["type"] <= 1.0:
    new_layer = layers.Dense(hidden_layer_definitions[0]["neurons"], activation_type)(inputs)
    hidden_layers.append(new_layer)
    
# If they exist, link additional hidden layers in sequence
if len(hidden_layer_definitions) > 1:
    for i in range(1, len(hidden_layer_definitions)):
        activation_type = get_activation_function_keyword(hidden_layer_definitions[i]["activation"])
        if hidden_layer_definitions[i]["type"] >= 0.0 and hidden_layer_definitions[i]["type"] <= 1.0:
            new_layer = layers.Dense(hidden_layer_definitions[i]["neurons"], activation_type)(hidden_layers[i - 1])
            hidden_layers.append(new_layer)    




# for layer in range(hidden_layers_count):
#     # determine the keyword for the activation type
#     activation_type = get_activation_function_keyword(layer["activation"])  

#     layer_def = layers.Dense(512, activation="relu")

# layer_out = layer_def(inputs)


# Create the output layer
# determine the keyword for the activation type
activation_type = get_activation_function_keyword(layer_output["activation"])       

if layer_output["type"] >= 0.0 and layer_output["type"] <= 1.0:
    outputs = layers.Dense(layer_output["count"], activation=activation_type)(hidden_layers[len(hidden_layers) - 1])
# outputs = layers.Dense(3, activation="linear")(layer1_out)


nn_model = keras.Model(inputs=inputs, outputs=outputs)




# this generates a command line output showing that the network exists and its design
# nn_model.summary()

# generate some input for the network, and run that through the network for a result
# input = keras.backend.constant([[0.5]])
# output = nn_model(input)
# print(output)

# generate a series of inputs between 0 and 1
# inputs = np.arange(0, 1.1, 0.1)
# print(inputs)

# inputs_list = inputs.tolist()
# print(inputs_list)

# for inp in inputs_list:
#     input = keras.backend.constant([[inp]])
#     output = nn_model(input)
#     print(output)
    
    
# get the weights of the network
print("\n\n\n###################")
for layer in nn_model.layers:
    # print(layer.get_config(), layer.get_weights())
    print(layer.get_config())
    print("\n\n")

# weights1 = layer1_def.get_weights()
# print(weights1)