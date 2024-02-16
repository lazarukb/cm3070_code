
# for storing the information for the generated games
class TextworldGames():
   
    def __init__(self):
        self.path = "tw_games"
        self.games_and_paths = {
            "2-3-10-v1": {"filename": "game_2-3-10-v1.ulx", "max_steps": 25},
            "coin_collector_5": {"filename": "tw-coin_collector-mKimsM-house-mo-010Gsa0Qf6BWHNep.ulx", "max_steps": 150},
            "coin_collector_15": {"filename": "tw-coin_collector-o0i5F0-house-mo-EKBgf9N2TYBPs3mJ.ulx", "max_steps": 450}
        }
        
        # re-write this into one that reads from a csv
        # and write something separately that creates world and writes the names and maximum steps to that csv
        
    
    def getGamePath(self, code):
        file_path = None
        file_path = self.path + "/" + self.games_and_paths[code]["filename"]
        return file_path
    
    def getGameMaxSteps(self, code):
        return self.games_and_paths[code]["max_steps"]