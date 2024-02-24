"""Runs the process of sending the networks of a Population through TextWorld.

The Simulation interfaces with TextWorld, to determine which are the valid steps,
 creates and updates the input state tensor for each step, evaluates the network's
 performance and applies rewards and penalties, finally assigning the network's
 fitness back to the Network instance for future use.
"""

import keras
import numpy as np
import random
import unittest
from copy import deepcopy

import textworld
import textworld.gym
import TextworldGames

# from keras import layers

class Simulation(unittest.TestCase):
    """Runs the process of sending the networks of a Population through TextWorld.

    The Simulation interfaces with TextWorld, to determine which are the valid steps,
    creates and updates the input state tensor for each step, evaluates the network's
    performance and applies rewards and penalties, finally assigning the network's
    fitness back to the Network instance for future use.

    Attributes:
    remove this: only need to be here if there is an __init__
        env_parameters: A DICT of which TextWorld environment parameters should
         be returned when TextWorld is queried for them.
    """
    
    def __init__(self):
        # self.simulation_id = 0
        self.env_parameters = textworld.EnvInfos(
            admissible_commands = True,
            entities = True,
            verbs = True,
            command_templates = True,
            moves = True,
            won = True,
            score = True
        )
    
    def registerEnvId(self, code):
        """Creates the Gymnasium environment, attached to a specified TextWorld game.

        As necessary, expand here.
        remove this: A docstring should give enough information to write a call to the function without reading the function’s code. The docstring should describe the function’s calling syntax and its semantics, but generally not its implementation details, unless those details are relevant to how the function is to be used.

        Args:
            code: str shortcode to identify which TextWorld map should be used.

        Returns:
            environment_id: the derived Gym ID object.
            max_steps: the INT maximum number of steps for the game
        """
        
        # Initialise the TextWorld games library, and read from it.
        
        tw_game_index = TextworldGames.TextworldGames()
        self.assertIsNotNone(code)
        max_steps = tw_game_index.getGameMaxSteps(code)
        file_path = tw_game_index.getGamePath(code)
        self.assertIsNotNone(file_path)
        
        # Pass the arguments to Gym to build the TextWorld/Gym ID for this game.
        
        environment_id = textworld.gym.register_game(
            file_path,
            self.env_parameters,
            max_episode_steps = max_steps
            )
        self.assertIsInstance(environment_id, str)
        
        return environment_id, max_steps
    
      
    def apply_nn_to_textworld(
        self,
        nn_obj,
        game,
        force_random_choice,
        force_pickup,
        steps_to_retain,
        failed_step_reward,
        valid_step_reward,
        chain_rewards
        ):
        """Runs the neural network through the TextWorld game.

        Args:
            nn_obj: the Keras neural network, as an object.
            game: INT the ID of the TextWorld/Gym environment to be used.
            force_random_choice: Boolean to force random movement if a the network
             is repeating a previously failed action.
            force_pickup: Boolean to force the network to choose to pick up an
             available coin, thus winning the game.
            steps_to_retain: INT number of previous steps to be included 
             in the input tensor for each new step.
            failed_step_reward: INT reward for making an invalid choice.
            valid_step_reward: INT reward for making a valid choice.
            chain_rewards: Boolean to sum previous valid step rewards, or not.

        Returns:
            INT of the fitness of this network, defined as the number of max_steps
             remaining when the network found the coin.
            
        """
        
        environment_id, max_steps = self.registerEnvId(game)
        environment = textworld.gym.make(environment_id)
        obs, infos = environment.reset()
        self.assertIsInstance(obs, str)
        self.assertIsInstance(infos, dict)

        # Per TextWorld docs, entities are everything in the game,
        #  anywhere in the maze.
        # Admissible commands are all the commands relevant to the current
        #  game state. So this will change from step to step.
        # Verbs - all as understood by the game, is static from step to step.
        # To build the action space of available commands that could be
        #  entered into the interpreter:
        #   Get the verbs, and remove the ones that aren't necessary here
        #   like drop, examine, inventory, and look.
        #   Which leaves an action space of go * 4, one for each cardinal
        #   direction, and take coin, since that's the only object in the game.
        
        action_space = ["take coin", "go east", "go west", "go north", "go south"]

        # Initialise the network's chosen action
        
        nn_action = None
        
        # Setup the previous_action_spaces_and_choices to the right size for
        #  storing the prescribed number of previous action spaces.
        
        previous_action_spaces_and_choices = []
        for i in range(steps_to_retain):
            previous_action_spaces_and_choices.append([0, 0, 0, 0, 0, 0, 0])
            
      
        # Here the network is actually playing the game.
        # This is where we feed the action space to the network and get
        #  a result, and assign that result to step.
        
        for i in range(max_steps, 0, -1):
            # Build the action space for this room/step - the curated list of
            #  actions the network could legally make from the list of 
            #  available actions in this room.

            # Poll the admissible actions - all the commands relevant to the 
            #  current game state.
            # Parse through the curated action space, and mark as to if it's 
            #  also in the currently admissible commands.
            
            # For each step, build the input tensor from a one-hot list of
            #  which of the global permitted commands are valid in this room.
            # So if there is no coin in the room, despite "take coin" being
            #  globally valid, the action space for this room would have a 0 for
            #  "take coin". The one-hot list for the steps in this room are
            #  stored in action_space_values[].
            
            action_space_values = []
            for action in action_space:
                if action in infos["admissible_commands"]:
                    action_space_values.append(1)
                else:
                    action_space_values.append(0)
                    
            # Combine the current action space with the previous action
            #  spaces and results, to build the full input state.
            
            final_input = deepcopy(action_space_values)
            for prev in previous_action_spaces_and_choices:
                for ele in prev:
                    final_input.append(ele)

            # Make sure the input is of the proper length.
            
            assert len(final_input) == ((steps_to_retain * 7) + 5), \
                "Input space is of the wrong length."

            # Convert the input space to a tensor, and feed that to the network
            #  to get the probabilities for each of the 5 possible actions.
            
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
            #  repeatedly walking into the same wall or picking up a coin
            #  that wasn't there.
            # To avoid this, check the current observation, which is the result
            #  of the last step in the game. If it suggests that there is no
            #  coin to pick up, which is the response when the network tries to
            #  pick up a coin that isn't there, or that the last step was
            #  invalid, remove those choices from the possible actions.
            # Basically, if the selected action is the same as the last action,
            #  then check if that did anything. If it failed, choose the next
            #  most likely option instead.
            # Else, set the action to be what the network has decided.
            
            # Use random choice instead of next most likely, if the flag is set
            
            if descending[0] == nn_action and force_random_choice:
                # The current step choice is the same as the previous.
                # Check what happened the last time this step was taken
                
                if previous_action_spaces_and_choices[0][6] == failed_step_reward:
                    # Means last step resulted in walking into a wall
                    #  or trying to pick up a coin that wasn't there.
                    # Avoid doing that again by choosing a different action.
                    
                    nn_action = descending[random.randint(1, len(descending) - 1)]
            else:
                # The last step didn't fail outright, accept network's choice
                #  to try it again.
                
                nn_action = descending[0]
                
            # If flagged, force the network to pick the coin if it's available,
            #  denoted by a action_space_values[0] == 1    
                    
            if action_space_values[0] == 1 and force_pickup:
                # action 0 is defined above as "take coin" - force it now since
                #  there is a coin in the room and the flag is true
            
                nn_action = 0
                
            # Now, having determined which step the network would like to take, 
            #  or overriding and forcing the decision, 
            #  send that action to the game.
            
            obs, score, done, infos = environment.step(action_space[nn_action])

            # Evaluate the step.
            # Check the info output to see if the network solved the game.
            #  If so, break out now. It's a Boolean, returning True|False.
            
            if infos['won']:
                break
            
            # Determine if the step resulted in running into a wall or trying to
            #  pick up a coin that wasn't there. 
            # After this we know the current action space, the step decision, 
            #  and the result of the step.
            # So populate that into the previous choices array for the next step.
            
            result_to_be_added = None
            if obs.strip() == "You can't go that way." or obs.strip() == "You can't see any such thing.":
                result_to_be_added = failed_step_reward
            else:
                # Here is a good step. The network didn't win the game, but also
                #  didn't walk into a wall or try to pick up a coin that wasn't
                #  there.
                # Check if we are to chain the rewards, incrementing them as a
                #  series of valid decisions are made, or not.
                
                if chain_rewards:
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
        

    def evaluate_population(
        self,
        sim_population,
        _game,
        _force_random_choice,
        _force_pickup,
        _steps_to_retain,
        failed_step_reward,
        valid_step_reward,
        chain_rewards
        ):
        """Evaluates the Population by sending the Networks through TextWorld.

        Takes the Population, and one at a time gets each Network from the
         Population and runs it through the TextWorld game, then commanding the
         Population to store the returned fitness of the Network.
         
        Args:
            sim_population: the Population instance to be evaluated.
            _game: INT the ID of the TextWorld/Gym environment to be used.
            _force_random_choice: Boolean to force random movement if a the network
             is repeating a previously failed action.
            _force_pickup: Boolean to force the network to choose to pick up an
             available coin, thus winning the game.
            _steps_to_retain: INT number of previous steps to be included 
             in the input tensor for each new step.
            failed_step_reward: INT reward for making an invalid choice.
            valid_step_reward: INT reward for making a valid choice.
            chain_rewards: Boolean to sum previous valid step rewards, or not.

        Returns:
            None. Writes back to the Population object and through it to the 
             Network objects in the Population object.
        """
        
        # Get the number of networks in the population
        # For each network, retrieve its Keras model, and push it through Gym
        # When this is called, the population of Network objects already exists
        #  as sim_population, which is a Population instance.
        # Get the number of networks in the population.
        
        networks_count = sim_population.get_population_size()
        for i in range(networks_count):
            # Get the Keras model from the Network object
            
            nn_obj = sim_population.get_neural_network_model(i)
            
            # Send each network off to play the game now, and 
            #  retrieve and store the fitness the Network scores.
            
            fitness = self.apply_nn_to_textworld(
                nn_obj,
                _game,
                _force_random_choice,
                _force_pickup,
                _steps_to_retain,
                failed_step_reward,
                valid_step_reward,
                chain_rewards
                )
            sim_population.set_nn_fitness(i, fitness)
            

        # Validation tests and console output ... good to show computing progress.
        fitnesses = []
        for i in range(networks_count):
            fitnesses.append(sim_population.get_nn_fitness(i))
        
        # Beginning of stats to the screen.
        print(f"Network count: {len(fitnesses)}.\t", end = "")
        print(f"Average fitness: {round(sum(fitnesses) / len(fitnesses), 2)}.\t", end = "")
        print(f"Max fitness: {str(max(fitnesses)).zfill(3)}.\t", end = "")
        