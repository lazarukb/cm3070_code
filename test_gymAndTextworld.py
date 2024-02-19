# Install textworld prerequisites and textworld itself
# https://github.com/microsoft/TextWorld#readme

# https://github.com/microsoft/TextWorld/blob/main/notebooks/Playing%20TextWorld%20generated%20games%20with%20OpenAI%20Gym.ipynb

# all the textworld env stuff: https://textworld.readthedocs.io/en/stable/textworld.html#textworld.core.EnvInfos

# At this point, since the creation of the games is not required to be done on every instantiation, the games are created by hand through manual commands being issued to the terminal.
# Each game is then attached to by the gym environment.

# unittesting, remember
# every function named "test"anything will be RUN by the unittest.main() command, in the order in which they appear in the script

# Essentially a proof of concept here.


import unittest
import textworld.gym
import textworld
import TextworldGames
import random

# !pip3 install gymnasium
# !pip3 install textworld

class testTextWorld(unittest.TestCase):
    def registerEnvId(self, code, env_parameters):
        print("testRegisterEnvId")
        tw_game_index = TextworldGames.TextworldGames()
        # maximum number of steps, basically a peak fitness network
        max_steps = tw_game_index.getGameMaxSteps(code)
        self.assertIsNotNone(code)
        file_path = tw_game_index.getGamePath(code)
        self.assertIsNotNone(file_path)
        environment_id = textworld.gym.register_game(file_path, env_parameters, max_episode_steps = max_steps)
        self.assertIsInstance(environment_id, str)
        return environment_id, max_steps
        
    # def makeGym(self, environment_id):
    #     print("testMakeGym")
        
    def testMain(self):
        print("main")
        
        # setup the parameters for the textworld
        env_parameters = textworld.EnvInfos(
            admissible_commands = True,
            entities = True,
            verbs = True,
            command_templates = True,
            moves = True,
            won = True,
            score = True
        )
        
        # tw-make tw-coin_collector --level 5 --format ulx
    
                
        environment_id, max_steps = self.registerEnvId("coin_collector_15", env_parameters)
        environment = textworld.gym.make(environment_id)
        obs, infos = environment.reset()
        self.assertIsInstance(obs, str)
        self.assertIsInstance(infos, dict)
        
        # Everything in the game, anywhere
        print("Entities: {}\n".format(infos["entities"]))
        
        # The admissible commands are all the commands relevant to the current game state. So this will change from step to step.
        print("Admissible commands:\n  {}".format("\n  ".join(infos["admissible_commands"])))
        
        # Verbs - all understood by the game, is static from step to step.
        print("Verbs:\n  {}".format("\n  ".join(infos["verbs"])))
        
        # Command templates - all understood by the game, static
        print("command_templates:\n  {}".format("\n  ".join(infos["command_templates"])))
        
        print("moves: {}\n".format(infos["moves"]))
        
        print("won: {}\n".format(infos["won"]))

        # To build the action space of available commands that could be entered into the interpreter
        # Get the verbs, and remove the ones that aren't necessary here like drop, examine, inventory, and look
        # Which leaves us with go * 4, one for each cardinal direction, and take coin, since that's the only object in the game.
        
        action_space = ["go east", "go west", "go north", "go south", "take coin"]
        
        # Initial state of the game
        print(obs)
        
        # Take some steps
        for i in range(max_steps, 0, -1):
            if "take coin" in infos["admissible_commands"]:
                print(f"845028734580927345089273458097238095728093745809273589072348957238905728093758927345872")
                print(f"Coin available to be taken")
            step = random.randint(0, len(action_space) - 1)
            print(f"\nFitness: {i}, command: {action_space[step]}")
            obs, score, done, infos = environment.step(action_space[step])
            print(obs)
            print(f"status {done}, {infos['moves']}, {infos['won']}")
            if done:
                break
        
        print("-=-=-==-=-=-=-=-=-=-=-=-=-==-==-")
        print(f"status {done}, {infos['moves']}, {infos['won']}")
        print(f"This network's fitness is {i - 1}")
        
        print("All done")
        # So the problem is that I need to have a static list of admissible commands
        

        
        

unittest.main()
