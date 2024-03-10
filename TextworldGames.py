"""Holds the locations and parameters of the maze games.

Games are made with a command like
 tw-make tw-coin_collector --level 5 --format ulx
"""

class TextworldGames():
    def __init__(self):
        self.path = "tw_games"
        self.games_and_paths = {
            "2-3-10-v1": {"filename": "game_2-3-10-v1.ulx", "max_steps": 25},
            "coin_collector_5": {"filename": "tw-coin_collector-mKimsM-house-mo-010Gsa0Qf6BWHNep.ulx", "max_steps": 150},
            "coin_collector_15": {"filename": "tw-coin_collector-o0i5F0-house-mo-EKBgf9N2TYBPs3mJ.ulx", "max_steps": 450},
            "coin_collector_50": {"filename": "tw-coin_collector-yjiNjCMb-house-mo-OEXXir9RFGXqSN1q.ulx", "max_steps": 1500}
        }
        """Holds the locations and parameters of the maze games.

        Games are made with a command like
        tw-make tw-coin_collector --level 5 --format ulx
        """    
    
    def get_game_path(self, code):
        """Getter to build and return the full path to a game on the filesystem.
    
        Args:
            code: shortcode of the game to be returned. 

        Returns:
            String of the full relative filepath to the game data file.
        """
        
        file_path = None
        file_path = self.path + "/" + self.games_and_paths[code]["filename"]
        return file_path
    
    def get_game_max_steps(self, code):
        """Getter to return the maximum steps to be played in a game.
    
        Args:
            code: shortcode of the game to be returned. 

        Returns:
            Integer of the arbitrary maximum number of steps allowed for that game.
        """
        
        return self.games_and_paths[code]["max_steps"]