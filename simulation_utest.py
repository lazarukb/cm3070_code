# import population
import keras
import numpy as np
import random
import unittest
from copy import deepcopy

import textworld
import textworld.gym
import TextworldGames

# tw-make tw-coin_collector --level 5 --format ulx

# from keras import layers

class Simulation(unittest.TestCase):
    def __init__(self):
        self.simulation_id = 0
        # setup the parameters for the textworld
        self.env_parameters = textworld.EnvInfos(
            admissible_commands = True,
            entities = True,
            verbs = True,
            command_templates = True,
            moves = True,
            won = True,
            score = True
        )
        
        
    # def create_random_population(self, number_of_networks):
    #     # the simulation should be creating the initial population and controlling the eventual population
    #     # which means moving that stuff from the test_and_run.py to here
    #     pass
    
    def registerEnvId(self, code):
        tw_game_index = TextworldGames.TextworldGames()
        # maximum number of steps, basically a peak fitness network
        self.assertIsNotNone(code)
        max_steps = tw_game_index.getGameMaxSteps(code)
        file_path = tw_game_index.getGamePath(code)
        self.assertIsNotNone(file_path)
        environment_id = textworld.gym.register_game(file_path, self.env_parameters, max_episode_steps = max_steps)
        self.assertIsInstance(environment_id, str)
        return environment_id, max_steps
    
      
    def apply_nn_to_textworld(self, nn_obj, game, force_random_choice, force_pickup, steps_to_retain, failed_step_reward, valid_step_reward, chain_rewards):
        environment_id, max_steps = self.registerEnvId(game)
        environment = textworld.gym.make(environment_id)
        obs, infos = environment.reset()
        self.assertIsInstance(obs, str)
        self.assertIsInstance(infos, dict)

        # Per TextWorld docs, entities are everything in the game, anywhere in the maze.
        #  The admissible commands are all the commands relevant to the current game state. So this will change from step to step.
        #  Verbs - all understood by the game, is static from step to step.
        #  To build the action space of available commands that could be entered into the interpreter
        #  Get the verbs, and remove the ones that aren't necessary here like drop, examine, inventory, and look
        #  Which leaves an action space of go * 4, one for each cardinal direction, and take coin, since that's the only object in the game.
        action_space = ["take coin", "go east", "go west", "go north", "go south"]

        # Initialise the network's chosen action
        nn_action = None
        
        # Setup the previous to zeroes and the current space will be prepended.
        previous_action_spaces_and_choices = []
        for i in range(steps_to_retain):
            previous_action_spaces_and_choices.append([0, 0, 0, 0, 0, 0, 0])
            
      
        # Here the network is actually playing the game.
        # This is where we feed the action space to the network and get a result, and assign that result to step
        for i in range(max_steps, 0, -1):
            
            # So get the possible states from the admissible commands statement
            # Then convert the action space to 0 if the command is not admissible, and 1 if it is.

                
            # Build the action space for this room/step - the curated list of helpful
            #  actions the network could take from the list of available actions in this room.

            # Poll the admissible actions - all the commands relevant to the current game state.
            # Parse through the curated action space, and mark as to if it's also in the currently admissible commands
            action_space_values = []
            for action in action_space:
                if action in infos["admissible_commands"]:
                    action_space_values.append(1)
                else:
                    action_space_values.append(0)
                    
            # Combine the current action space with the previous action spaces and results
            final_input = deepcopy(action_space_values)
            for prev in previous_action_spaces_and_choices:
                for ele in prev:
                    final_input.append(ele)

            # self.assertEqual(len(final_input), (steps_to_retain * 7) + 5)

            # Convert the input space to a tensor, and feed that to the network
            #  to get the probabilities for each of the 5 actions.
            nn_input_tensor = keras.backend.constant([final_input])
            nn_outputs = nn_obj(nn_input_tensor)
                      
            # Use argsort to determine the arrangement of the probabilities
            #  in the returned list. This is required over argmax because there
            #  is a chance that the most preferred step will be discarded.
            sorted = np.argsort(nn_outputs[0].numpy())
            descending = sorted[::-1]
            nn_action = descending[0]
            
            # Here is the editing out a choice which is proven to not work.
            # In early testing the networks were very likely to get stuck, 
            #  continually walking into the same wall or picking up a coin that wasn't there.
            # To avoid this, check the current observation, which is the result
            #  of the last step in the game. If it suggests that there is no
            #  coin to pick up, which is the response when the network tries to
            #  pick up a coin that isn't there, or that the last step was
            #  invalid, remove those choices from the possible actions.
            # Basically, if the selected action is the same as the last action,
            #  then check to see if that did anything. If it failed, choose the next
            #  most likely option instead.
            # Else, set the action to be what the network has decided.
            
            # Use a random choice instead of the next most likely, if the flag is set
            if descending[0] == nn_action and force_random_choice:
                # The current step choice is the same as the previous.
                # Check what happened the last time this step was taken
                if previous_action_spaces_and_choices[0][6] == failed_step_reward:
                    # A 0 here means the last step resulted in walking into a wall
                    #  or trying to pick up a coin that wasn't there.
                    # So, avoid doing that again by randomly choosing a different action.
                    nn_action = descending[random.randint(1, len(descending) - 1)]
            else:
                # The last step didn't fail outright, so accept the network's choice
                #  to try it again.
                nn_action = descending[0]
                
            # If flagged, force the network to pick the coin if it's available,
            #  denoted by a action_space_values[0] == 1            
            if action_space_values[0] == 1 and force_pickup:
                # action 0 is defined above as "take coin" - force it now since
                #  there is a coin in the room and the flag is true
                nn_action = 0
                
            # Now, take the step through the game    
            obs, score, done, infos = environment.step(action_space[nn_action])

            # Evaluate the step.
            # Check the info output to see if the network solved the game.
            #  If so, break out now.
            if infos['won'] == True:
                break
            
            # Determine if the step resulted in running into a wall or trying to
            #  pick up a coin that wasn't there. 
            # After this we know the current action space, the step decision, and the result of the step.
            # So populate that into the previous choices array for the next step.
            result_to_be_added = None
            if (obs.strip() == "You can't go that way.") or (obs.strip() == "You can't see any such thing."):
                result_to_be_added = failed_step_reward
            else:
                # Here is a good step. The network didn't win the game, but also
                #  didn't walk into a wall or try to pick up a coin that wasn't
                #  there.
                # Check if we are to chain the rewards, incrementing them as a
                #  series of valid decisions are made, or not.
                if chain_rewards:
                    # Increment the reward from the previous step.
                    previous_step_reward = previous_action_spaces_and_choices[len(previous_action_spaces_and_choices) - 1][6]
                    result_to_be_added = valid_step_reward + previous_step_reward
                    assert (result_to_be_added == (valid_step_reward + previous_step_reward))
                else:
                    result_to_be_added = valid_step_reward
            
            # We have the current action space, the step decision, and the 
            #  result of the step. Populate that into the previous choices
            #  array for the next step/iteration of this loop.
            self.assertIsNotNone(result_to_be_added)
            this_step_and_results = deepcopy(action_space_values)
            this_step_and_results.append(nn_action)
            this_step_and_results.append(result_to_be_added)   
            previous_action_spaces_and_choices.append(this_step_and_results)

            # Finally, with the previous results now appended with the most recent
            #  step choices and results, remove the oldest entry in the list,
            #  to keep it at the required number of input steps.
            previous_action_spaces_and_choices.pop(0)
            
        # And return the fitness of this network. If the loop broke then i will
        #  be greater than 1, representing the number of steps still available
        #  when decrementing from the max_steps. If the network failed to find
        #  the coin then i will be 1, the minimum fitness.
        return i
        

    def evaluate_population(self, sim_population, _game, _force_random_choice, _force_pickup, _steps_to_retain, failed_step_reward, valid_step_reward, chain_rewards):
        # Get the number of networks in the population
        # For each network, retrieve its Keras model, and push it through the gym
        # When this is called, the population of Network objects already exists as sim_population, which is a Population instance
        # Get the number of networks in the population
        networks_count = sim_population.get_population_size()
        for i in range(networks_count):
            # Get the Keras model from the Network object
            nn_obj = sim_population.get_neural_network_model(i)
            
            # Send each network off to play the game now, and eventually this should return the fitness value
            fitness = self.apply_nn_to_textworld(nn_obj, _game, _force_random_choice, _force_pickup, _steps_to_retain, failed_step_reward, valid_step_reward, chain_rewards)
            sim_population.set_nn_fitness(i, fitness)
            

        # Validation tests
        fitnesses = []
        for i in range(networks_count):
            fitnesses.append(sim_population.get_nn_fitness(i))
        
        # Beginning of stats to the screen.
        print(f"Network count: {len(fitnesses)}. ", end = "")
        print(f"Average fitness: {sum(fitnesses) / len(fitnesses)}. ", end = "")
        print(f"Max fitness: {max(fitnesses)}")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # Below here, all junk. Just for notes for now.
        '''
        
        for i in range(networks_count):
            # Get each of the network definitions
            # network_def = population.Population.get_neural_network_def(sim_population, i)
            network_def = sim_population.get_neural_network_def(sim_population, i)
            
            # Define the input layer from the dna
            inputs = layers.Input(shape = (network_def["input"],))
            
            # Get the definitions for the hidden layers
            hidden_layer_definitions = network_def["hidden_layers"]
            
            # print(hidden_layer_definitions[0]["activation"])
            
            # Create the hidden layers
            hidden_layers = []

            # print(hidden_layer_definitions[0]["neurons"])

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
            activation_type = self.get_activation_function_keyword(network_def["output"]["activation"])

            # Create and link the output layer to the last hidden layer
            if network_def["output"]["type"] >= 0.0 and network_def["output"]["type"] <= 1.0:
                outputs = layers.Dense(network_def["output"]["count"], activation=activation_type)(hidden_layers[len(hidden_layers) - 1])
    
    
            # Build the model
            print("Building neural network " + str(i))
            nn_model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Gather the weights and biases of the layers, input, all hidden, and output layers are included in the method. The first is the input, which has no weights, so skip it.
            for layer in range(1, len(nn_model.layers) - 0):
                weights_biases = (nn_model.layers[layer].get_weights())
                layer_config = (nn_model.layers[layer].get_config())
                # print("Weights as the model is generated")
                # print(type(weights_biases), str(len(weights_biases)))
                # nn_model.layers[layer].set_weights(weights_biases)
                population.Population.save_weight_bias_definitions(sim_population, i, layer, weights_biases)
                # print("weights retrieved from the network object")
                # print(population.Population.get_weight_bias_definitions(sim_population, i, layer))
                
                # print(nn_model.layers)
                
                
                # if i == 2:
                #     print("\n\n")
                #     print("Output to see the weights_biases and config data")
                #     print("Print for only one network")
                #     print("Layer: " + str(layer))
                #     print(weights_biases)
                #     print("\n\n")
                #     print(layer_config)
                #     print("\n\n")
                
                    
                
        '''
            
            # This needs to be cleaned up. The simulation needs to control the population and that needs to interface with the network objects ... most of the stuff above should be in network.py and just the definitions or the network object (keras.Model) should be returned from network.py
            
            
            # Run the model through the gym and collect its fitness
            # e.action_space contains the possible commands for the gym
            # Get the initial step into the gym, e.reset(), e.step(1) which returns a 4-tuple: (state,action,done,info)or (observation, reward, done, info)
            # e.step(action) is what we need to determine. Which action? That's for the nn to determine
            # also be sure to check what the done state is again.
            # Had to force cffi version matching to 1.15 -> https://stackoverflow.com/questions/68592184/error-version-of-cffi-python-package-mismatch
            
            # sudo apt install python3-cffi    and then   pip install --upgrade cffi==1.15.0  (because the first command installed 1.15.0)
            
            # After that ... textworld installed!
            
            
            # Loop over parsing the offered action space
            
            
            # Gather fitness scores
            #################### temporary
            #################### artificial fitness scores
            #################### need to be normalised to 0-1
            
            # population.Population.set_nn_fitness(sim_population, i, random.random())
            
            
            
            
            
            
            
    # def get_activation_function_keyword(self, _activation):
    #     ''' Takes input of the float value of the activation gene, returns the keyword relating to that activation function'''
    #     match _activation:
    #         case _activation if 0.0 <= _activation < 0.33:
    #             activation_type = "relu"
    #         case _activation if 0.33 <= _activation < 0.67:
    #             activation_type = "linear"
    #         case _activation if 0.67 <= _activation <= 1.0:
    #             activation_type = "sigmoid"
    #     return activation_type    
    
# unittest.main()

    # def run_network(self, population, max_steps = 50):


'''
    
      
    def apply_nn_to_textworld(self, nn_obj, game, force_random_choice, force_pickup, steps_to_retain):
        environment_id, max_steps = self.registerEnvId(game)
        environment = textworld.gym.make(environment_id)
        obs, infos = environment.reset()
        self.assertIsInstance(obs, str)
        self.assertIsInstance(infos, dict)
        
        # Everything in the game, anywhere
        # print("Entities: {}\n".format(infos["entities"]))
        
        # The admissible commands are all the commands relevant to the current game state. So this will change from step to step.
        # print("Admissible commands:\n  {}".format("\n  ".join(infos["admissible_commands"])))
        
        # Verbs - all understood by the game, is static from step to step.
        # print("Verbs:\n  {}".format("\n  ".join(infos["verbs"])))
        
        # Command templates - all understood by the game, static
        # print("command_templates:\n  {}".format("\n  ".join(infos["command_templates"])))
        
        # print("moves: {}\n".format(infos["moves"]))
        
        # print("won: {}\n".format(infos["won"]))

        # To build the action space of available commands that could be entered into the interpreter
        # Get the verbs, and remove the ones that aren't necessary here like drop, examine, inventory, and look
        # Which leaves us with go * 4, one for each cardinal direction, and take coin, since that's the only object in the game.
        
        # moved take coin to the first selection as I think the networks are favouring the first choice, since it works in the first step of the game.
        
        action_space = ["take coin", "go east", "go west", "go north", "go south"]
        nn_action = None
        # action_space_values = [[0, 1, 1, 0, 0]]
        
        # Initial state of the game
        # print(f"Initial state of the game")
        # print(obs)
        
        # Setup the previous to zeroes and the current space will be prepended. So total inputs will equal steps * 7 + 5.
        # steps_to_retain = 5
        previous_action_spaces_and_choices = []
        for i in range(steps_to_retain):
            previous_action_spaces_and_choices.append([0, 0, 0, 0, 0, 0, 0])
            
        # self.assertEqual(len(previous_action_spaces_and_choices), steps_to_retain * 7)
        # previous_results = [0]
        
        # Take some steps
        for i in range(max_steps, 0, -1):
            # step = random.randint(0, len(action_space) - 1)
            # This is where we feed the action space to the network and get a result, and assign that result to step
            # This is where everything ties together.
            # Right here.
            
            # So get the possible states from the admissible commands statement
            # Then convert the action space to 0 if the command is not admissible, and 1 if it is.
            # Add the current number of steps as well?
            # Make that a tensor and fire it into the network
            # and reconfigure the network to provide 5 responses and then do an argmax on them?
            
            
            # random input action space for testing
            # action_space_values = []
            # for j in range(5):
            #     action_space_values.append(random.randint(0,1))
                
                
            # Poll the admissible actions for this space
            # The admissible commands are all the commands relevant to the current game state. So this will change from step to step.
            # print("Admissible commands:\n  {}".format("\n  ".join(infos["admissible_commands"])))
            
            # Parse through the curated action space, and mark as to if it's also in the currently admissible commands
            action_space_values = []
            for action in action_space:
                if action in infos["admissible_commands"]:
                    action_space_values.append(1)
                else:
                    action_space_values.append(0)
                    
            # print(f"So the state or valid action_space is: {action_space_values}")
            
            # Now build the current action space in with the previous space, choice, and result of that choice.
            
            # action_space_and_prev_states = [[action_space_values], [0, 0, 0, 0, 0, 0, 0]]
            
            # Flatten the current action space with the previous action spaces and results
            final_input = deepcopy(action_space_values)
            
            for prev in previous_action_spaces_and_choices:
                for ele in prev:
                    final_input.append(ele)

            # print(f"The final input for this step: {final_input}")
            # print(f"Length of input: {len(final_input)}, target length of input: {(steps_to_retain * 7) + 5}")
                    
            # self.assertEqual(len(final_input), (steps_to_retain * 7) + 5)
                    
            # print(f"action_space_values: {action_space_values}")
            
            # nn_input_tensor = keras.backend.constant([action_space_values])
            nn_input_tensor = keras.backend.constant([final_input])
            # print(f"{nn_input_tensor}, {nn_input_tensor.shape}")
            
            # Here, the network is fed the inputs and returns the probabilities for each of the action space choices.
            nn_outputs = nn_obj(nn_input_tensor)
            
            # print (f"nn_outputs: {nn_outputs}")
            
            # Now just use argmax to select the index of the largest element in the output tensor - the nn's "choice"
            # argsort instead -- then compare the most liely against the previous and skip it if it previously failed.
            # nn_action = keras.backend.argmax(nn_outputs[0]).numpy()
            # nn_action = keras.backend.argsort(nn_outputs[0]).numpy()
            
            # print(f"type: {type(nn_outputs)}, data: {nn_outputs}")
            
            # Get the indices in order of value
            
            sorted = np.argsort(nn_outputs[0].numpy())
            descending = sorted[::-1]
            nn_action = descending[0]
            
            # print(f"sorted: {sorted}")
            # print(f"descending: {descending}")
            
            # print(f"Meaning that the network's selected action is {descending[0]}, and its second choice is {descending[1]}")
            
            # Now here we need to edit out a choice which is proven to not work.
            
            # Now, in early testing the networks were very likely to get stuck, continually walking into the same wall or picking up a coin that wasn't there.
            # To avoid this, we check the current obs. If it suggests that there is no coin to pick up, or that the last step was invalid, remove those choices from the possible actions
            # Slice the obs to disregard the newlines.
            # print("\n\n")
            # print(f"Last obs {obs.strip()}, last action: {nn_action}")
            
            # if the selected action is the same as the last action, then check to see if that did anything. If not, choose the next most likely option make instead a random selection from the remaining choices.
            # else, set the action to be what the network has decided.
            
            # force_random_choice = False
            if descending[0] == nn_action and force_random_choice:
                # The current step choice is the same as the previous.
                # if (obs.strip() == "You can't go that way.") or (obs.strip() == "You can't see any such thing."):
                # Check what happened the last time this step was taken
                if previous_action_spaces_and_choices[0][6] == 0:
                    # print(f"You can't go that way detected or You can't see a coin detected. A random choice will be taken instead.")
                    # nn_action = descending[1]
                    nn_action = descending[random.randint(1, len(descending) - 1)]
                    # print(action_space_values)
                    # action_space_values[nn_action] = 0
                    # print(f"before type: {type(nn_outputs)}, {nn_outputs}")
                    # nn_outputs[nn_action] = 0
                    # print(f"after type: {type(nn_outputs)}, {nn_outputs}")
                    # print(action_space_values)  
                # elif obs.strip() == "You can't see any such thing.":
                    # print(f"You can't see a coin detected")
                    # print(action_space_values)
                    # action_space_values[0] = 0
                    # print(action_space_values)
            else:
                nn_action = descending[0]
                
            # print(f"Network's choice was {descending[0]}, and the final decision step will be {nn_action}")
            

        
            # print(f"nn_action: {nn_action}")
            
            # Force the network to pick the coin if it's available            
            # force_pickup = True
            if action_space_values[0] == 1 and force_pickup:
                # print(f"-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-")
                # print(f"The coin is available: {action_space_values}")
                # print(f"Did the network pick it? {nn_action}, {action_space[nn_action]}")
                # print(f"Forcing the network to pick the coin")
                nn_action = 0
                # print(f"Action is now: {action_space[nn_action]}")
                # print(f"\nFitness: {i}, command: {action_space[nn_action]}")
                # print(f"Action being taken: {action_space[nn_action]}")
                
            
            # print(f"Current state: {obs}")
            # print(f"Taking step: {action_space[nn_action]}")
            
                
            
            # Now, take the step through the game    
            obs, score, done, infos = environment.step(action_space[nn_action])
                # print(obs)
                # print(f"-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-")
            # else:                
            #     # print(f"\nFitness: {i}, command: {action_space[nn_action]}")
            #     # print(f"Action being taken: {action_space[nn_action]}")
            #     obs, score, done, infos = environment.step(action_space[nn_action])
            # print(obs)
            # print(f"status {done}, {infos['moves']}, {infos['won']}")
            
            # print(f"Resulting state: {obs}")
            
            if infos['won'] == True:
                # print(f"-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-")
                # print(f"WE WIN!")
                # print(f"-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-")
                break
            
            # Determine if the step resulted in running into a wall or trying to pick up a coin that wasn't there
            # After this we know the current action space, the step decision, and the result of the step.
            # So populate that into the previous choices array for the next step.
            result_to_be_added = None
            if (obs.strip() == "You can't go that way.") or (obs.strip() == "You can't see any such thing."):
                this_step_and_results = deepcopy(action_space_values)
                this_step_and_results.append(nn_action)
                result_to_be_added = 0
                this_step_and_results.append(result_to_be_added)
                # print(f"Current step score: 0")
                # print(f"Walked into a wall or such. Result should be 0.")
                
            else:
                this_step_and_results = deepcopy(action_space_values)
                this_step_and_results.append(nn_action)
                reward_for_good_step = 1
                # Get the previous state score, to be added to
                # previous_step_state = previous_action_spaces_and_choices[len(previous_action_spaces_and_choices) - 1][6]
                # result_to_be_added = reward_for_good_step + previous_step_state
                result_to_be_added = reward_for_good_step
                this_step_and_results.append(result_to_be_added)
                # print(f"Current step score: {result_to_be_added}")
                # print(f"Traversed the maze! Result should be 1.")
                
            previous_action_spaces_and_choices.append(this_step_and_results)
            
            # self.assertEqual(1, 1)
            if result_to_be_added != previous_action_spaces_and_choices[len(previous_action_spaces_and_choices) - 1][6]:
                # print(f"These should be equal")
                quit()
            
            
            
            # print(f"{previous_action_spaces_and_choices}, ", end="")
            
            # Trim the previous results and spaces to the maximum intended to be retained by removing the oldest.
            previous_action_spaces_and_choices.pop(0)
            
            # print(f"{previous_action_spaces_and_choices}, ")
            
        
        # print("-=-=-==-=-=-=-=-=-=-=-=-=-==-==-")
        # print(f"status {done}, {infos['moves']}, {infos['won']}")
        # print(f"This network's fitness is {i - 1}")
        
        # print("All done")
        # So the problem is that I need to have a static list of admissible commands
        
        # And return the fitness of this network
        
        
        ## This is artifically returning a fitness of at minimum of 1 for now until this is properly handled, or if in fact this is decided to be the way of things
        
        return i
        

    
    def evaluate_population(self, sim_population, _game, _force_random_choice, _force_pickup, _steps_to_retain):
        # Get the number of networks in the population
        # For each network, retrieve its Keras model, and push it through the gym
        
        # When this is called, the population of Network objects already exists as sim_population, which is a Population instance
        # Get the number of networks in the population
        # networks_count = population.Population.get_population_size(sim_population)
        # self.assertEqual(1, 1)
        networks_count = sim_population.get_population_size()
        # print(f"Evaluating networks ", end = "")
        for i in range(networks_count):
            # Get the Keras model from the Network object
            # print(f".", end = "")
            nn_obj = sim_population.get_neural_network_model(i)
            # print(nn_obj.layers[1].get_weights())
            
            # Send each network off to play the game now, and eventually this should return the fitness value
            fitness = self.apply_nn_to_textworld(nn_obj, game = _game, force_random_choice = _force_random_choice, force_pickup = _force_pickup, steps_to_retain = _steps_to_retain)
            
            
            # Set the fitness of this network - for now ####################################################################
            # But here we will put the network through the gym and recover its fitness from there ##########################
            # sim_population.set_nn_fitness(i, random.random())
            
            # print(f"simulation: the network {i} scored a fitness of {fitness}")
            sim_population.set_nn_fitness(i, fitness)
            
            # print(f"Now let's get the fitness and see if it's there")
            # print(f"Network {i} reports serial number {population.Population.get_neural_network_def(sim_population, i)['meta']['serial_number']} and fitness of {sim_population.get_nn_fitness(i)}")
        
        # print(f"")
        fitnesses = []
        # Validation tests
        for i in range(networks_count):
            # print("simulation: Network: " + str(i) + " reports fitness of " + str(sim_population.get_nn_fitness(i)))
            fitnesses.append(sim_population.get_nn_fitness(i))
        # population.Population.set_nn_fitness(sim_population, i, random.random())
        
        # Beginning of stats
        print(f"Number of networks in this generation: {len(fitnesses)}. ", end = "")
        print(f"Average fitness of this generation: {sum(fitnesses) / len(fitnesses)}. ", end = "")
        print(f"Max fitness of this generation: {max(fitnesses)}")
        
        
        # Below here, all junk. Just for notes for now.

        for i in range(networks_count):
            # Get each of the network definitions
            # network_def = population.Population.get_neural_network_def(sim_population, i)
            network_def = sim_population.get_neural_network_def(sim_population, i)
            
            # Define the input layer from the dna
            inputs = layers.Input(shape = (network_def["input"],))
            
            # Get the definitions for the hidden layers
            hidden_layer_definitions = network_def["hidden_layers"]
            
            # print(hidden_layer_definitions[0]["activation"])
            
            # Create the hidden layers
            hidden_layers = []

            # print(hidden_layer_definitions[0]["neurons"])

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
            activation_type = self.get_activation_function_keyword(network_def["output"]["activation"])

            # Create and link the output layer to the last hidden layer
            if network_def["output"]["type"] >= 0.0 and network_def["output"]["type"] <= 1.0:
                outputs = layers.Dense(network_def["output"]["count"], activation=activation_type)(hidden_layers[len(hidden_layers) - 1])
    
    
            # Build the model
            print("Building neural network " + str(i))
            nn_model = keras.Model(inputs=inputs, outputs=outputs)
            
            # Gather the weights and biases of the layers, input, all hidden, and output layers are included in the method. The first is the input, which has no weights, so skip it.
            for layer in range(1, len(nn_model.layers) - 0):
                weights_biases = (nn_model.layers[layer].get_weights())
                layer_config = (nn_model.layers[layer].get_config())
                # print("Weights as the model is generated")
                # print(type(weights_biases), str(len(weights_biases)))
                # nn_model.layers[layer].set_weights(weights_biases)
                population.Population.save_weight_bias_definitions(sim_population, i, layer, weights_biases)
                # print("weights retrieved from the network object")
                # print(population.Population.get_weight_bias_definitions(sim_population, i, layer))
                
                # print(nn_model.layers)
                
                
                # if i == 2:
                #     print("\n\n")
                #     print("Output to see the weights_biases and config data")
                #     print("Print for only one network")
                #     print("Layer: " + str(layer))
                #     print(weights_biases)
                #     print("\n\n")
                #     print(layer_config)
                #     print("\n\n")
                
            
            # This needs to be cleaned up. The simulation needs to control the population and that needs to interface with the network objects ... most of the stuff above should be in network.py and just the definitions or the network object (keras.Model) should be returned from network.py
            
            
            # Run the model through the gym and collect its fitness
            # e.action_space contains the possible commands for the gym
            # Get the initial step into the gym, e.reset(), e.step(1) which returns a 4-tuple: (state,action,done,info)or (observation, reward, done, info)
            # e.step(action) is what we need to determine. Which action? That's for the nn to determine
            # also be sure to check what the done state is again.
            # Had to force cffi version matching to 1.15 -> https://stackoverflow.com/questions/68592184/error-version-of-cffi-python-package-mismatch
            
            # sudo apt install python3-cffi    and then   pip install --upgrade cffi==1.15.0  (because the first command installed 1.15.0)
            
            # After that ... textworld installed!
            
            
            # Loop over parsing the offered action space
            
            
            # Gather fitness scores
            #################### temporary
            #################### artificial fitness scores
            #################### need to be normalised to 0-1
            
            # population.Population.set_nn_fitness(sim_population, i, random.random())
            
'''