"""Gathers and writes out various reporting to CSV.
"""

import os
import shutil

class Reporting():
    """Gathers and writes out various reporting to CSV.
    """
    
    def census(sim_population):
        """Gathers the metadata and fitnesses of the passed Population.

        Creates the reporting data for a generation by gathering the metadata
         and associated assigned fitness scores for the networks in a Population.

        Args:
            sim_population: The Population instance being evaluated.

        Returns:
            A list of lists, each of the meta data and fitness for each network.
        """
        
        results = []
        population_size = sim_population.get_population_size()
        
        for i in range(population_size):
            nn_meta = sim_population.get_neural_network_def(i)
            nn_fitness = sim_population.get_nn_fitness(i)
            results.append([nn_meta, nn_fitness])
        
        return results
    
    
    def create_folders(parameters):
        """Creates the reporting folder structure on disk.

        If the experiment number is specified as 0 then no reporting to disk
         will be done and the method does not need to run, so it returns.
        If the experiment number is 1 then the existing folders are deleted
         so it can be reused. This is used during development.
         
        Args:
            parameters: DICT containing all the experimental settings. The 
             comment, experiment number, and subfolder are used here.

        Returns:
            None. Creates folder structure on disk.
            
        Raises:
            Will fail if the experiment folder already exists, to avoid
             overwriting existing experimental results.
        """
        
        comment = parameters['experiment_comment']
        maindir = parameters['experiment_number']
        subdir = parameters['subfolder']
        fulldir = maindir + "/" + subdir
        if subdir == 0 or maindir == 0:
            return
        
        # Make the report directory
        
        if maindir == 1 or subdir == 1:
            # remove the subdir and contents as this is the testing directory
            # intended for when the output should be created, not not retained.
            
            try:
                shutil.rmtree("experiments/" + str(fulldir))
            except:
                pass
        
        # First, the main experiment directory which needs to be skipped if it
        #  already exists, to accommodate the subdirectories. Quick warning to 
        #  the console, just to be sure.
        
        try:
            os.mkdir("experiments/" + str(maindir))
            os.mkdir("experiments/" + str(maindir) + "/!-- " + str(comment))
        except:
            print(f"\n\n ------------------------------------------------\n")
            print(f"Note, the main directory {maindir} already exists. " +
                "Subfolders are being mixed with new and previous data.")
            print(f"\n --------------------------------------------------\n")

        # But subfolders must not be overwritten if they exist as this is where
        #  the data is stored.
        
        try:
            os.mkdir("experiments/" + str(maindir) + "/" + str(subdir))
        except:
            print(f"\n\nFATAL. The directory {fulldir} already exists.\n\n")
            quit(1)
        
    
    def output_simulation_to_csv(parameters, report):
        """Outputs the large reporting dict to various CSV files.

        Performs fitness average calculations and stores the outputs of each
         network, each generation, and summary data.
        If the experiment number is specified as 0 then no reporting to disk
         will be done and the method does not need to run, so it returns.
         
        Args:
            parameters: DICT containing all the experimental settings. The 
             comment, experiment number, and subfolder are used here.
            report: DICT containing nested DICT reporting for the networks at
             various states

        Returns:
            Float of the average fitness of the entire simulation. This is
             gathered by the run_experiments.py to report on the average fitness
             returned by each individual set of experimental parameters.
        """
        
        maindir = parameters['experiment_number']
        subdir = parameters['subfolder']
        fulldir = maindir + "/" + subdir
        
        if subdir == 0 or maindir == 0:
            return
        
        # Write the experiment parameters
        
        keys = report['parameters'].keys()
        values = report['parameters'].values()
        with open("experiments/" + str(fulldir) + "/parameters.csv", 'w') as f:
            for key in keys:
                f.write(key + ",")
            f.write("\n")
            for value in values:
                f.write(str(value) + ",")
            f.write("\n")
       
        # Write the header and details for each network in each generation.
        
        with open("experiments/" + str(fulldir) + "/nn_and_results_data.csv", 'w') as f:
            header = (
                "generation",
                "stage",
                "#",
                "serial_number",
                "checksum",
                "parent_1",
                "parent_2",
                "hidden_checksum",
                "output_checksum",
                "input",
                "hidden_type",
                "hidden_neurons",
                "hidden_activation",
                "output_type",
                "output_count",
                "output_activation",
                "fitness"
                )
            for ele in header:
                f.write(ele + ",")
            f.write("\n")
            
            # Write the details for the initial population, with some filler.
            
            for nn in range(len(report['initial_population'])):
                f.write("initial, initial,")
                f.write(f"{nn},")
                for key in report['initial_population'][nn][0]['meta']:
                    f.write(f"{report['initial_population'][nn][0]['meta'][key]},")
                f.write(f"{report['initial_population'][nn][0]['inputs']},")
                for key in report['initial_population'][nn][0]['hidden_layers'][0]:
                    f.write(f"{report['initial_population'][nn][0]['hidden_layers'][0][key]},")
                for key in report['initial_population'][nn][0]['output']:
                    f.write(f"{report['initial_population'][nn][0]['output'][key]},")
                f.write(f"{report['initial_population'][nn][1]}\n")


        # Write the output for each network in each generation after evaluation.
        
        for gen in range(len(report['generations'])):
            with open("experiments/" + str(fulldir) + "/nn_and_results_data.csv", 'a') as f:
                # Write the after evaluation details
                for nn in range(len(report['generations'][gen]['after_evaluation'])):
                    f.write(f"{gen},after_evaluation,{nn},")
                    for key in report['generations'][gen]['after_evaluation'][nn][0]['meta']:
                        f.write(f"{report['generations'][gen]['after_evaluation'][nn][0]['meta'][key]},")
                
                    f.write(f"{report['generations'][gen]['after_evaluation'][nn][0]['inputs']},")
                    for key in report['generations'][gen]['after_evaluation'][nn][0]['hidden_layers'][0]:
                        f.write(f"{report['generations'][gen]['after_evaluation'][nn][0]['hidden_layers'][0][key]},")
                    for key in report['generations'][gen]['after_evaluation'][nn][0]['output']:
                        f.write(f"{report['generations'][gen]['after_evaluation'][nn][0]['output'][key]},")
                    f.write(f"{report['generations'][gen]['after_evaluation'][nn][1]}\n")
             
             
            # Write the output for each network after carry-over.   
    
            with open("experiments/" + str(fulldir) + "/nn_and_results_data.csv", 'a') as f:
                # Write the after carryover details
                for nn in range(len(report['generations'][gen]['after_carryover'])):
                    f.write(f"{gen},after_carryover,{nn},")
                    for key in report['generations'][gen]['after_carryover'][nn][0]['meta']:
                        f.write(f"{report['generations'][gen]['after_carryover'][nn][0]['meta'][key]},")
                
                    f.write(f"{report['generations'][gen]['after_carryover'][nn][0]['inputs']},")
                    for key in report['generations'][gen]['after_carryover'][nn][0]['hidden_layers'][0]:
                        f.write(f"{report['generations'][gen]['after_carryover'][nn][0]['hidden_layers'][0][key]},")
                    for key in report['generations'][gen]['after_carryover'][nn][0]['output']:
                        f.write(f"{report['generations'][gen]['after_carryover'][nn][0]['output'][key]},")
                    f.write(f"{report['generations'][gen]['after_carryover'][nn][1]}\n")

       
        # Write the summary for each generation.
  
        with open("experiments/" + str(fulldir) + "/generation_summary.csv", 'w') as f:
            header = ("generation","number_of_networks","maximum_fitness","average_fitness")
            for ele in header:
                f.write(ele + ",")
            f.write("\n")
        
        # Calculate average fitness of this simulation, to be returned and written.
        
        sim_avg_fitness = 0
        for gen in range(len(report['generations'])):
            # Calculate and write the output for each network in each
            #  generation, after evaluation
            
            fitnesses = []
            num_nn = len(report['generations'][gen]['after_evaluation'])
            
            # Gather metrics per nn
            
            for nn in range(len(report['generations'][gen]['after_evaluation'])):
                nn_fitness = report['generations'][gen]['after_evaluation'][nn][1]
                fitnesses.append(nn_fitness)
                
            # Summarise metrics
            
            max_fitness = max(fitnesses)
            avg_fitness = round(sum(fitnesses) / num_nn, 2)
            sim_avg_fitness += avg_fitness
            
            with open("experiments/" + str(fulldir) + "/generation_summary.csv", 'a') as f:
                f.write(f"{gen},{num_nn},{max_fitness},{avg_fitness}")
                f.write("\n")
                
        sim_avg_fitness = round(sim_avg_fitness / len(report['generations']),4)
        os.mkdir("experiments/" + str(fulldir) + "/!-- simulation average fitness -- " + str(sim_avg_fitness))
        
        return sim_avg_fitness


    def output_runtime_stats_to_csv(most_fit_folders, parameters, cumulative):
        """Outputs per experiment and per runtime statistics to CSV..
         
        Args:
            parameters: DICT containing all the experimental settings. The 
             comment, experiment number, and subfolder are used here.
            cumulative: DICT containing some runtime stats such as total
             elapsed time.

        Returns:
            None - outputs to disk.
        """
        
        with open("experiments/" + str(parameters['experiment_number']) + "/fitness_summary_by_experiment_" + cumulative['unique_stamp'] + ".csv", 'w') as f:
            header = ("subfolder","fitness",)
            for ele in header:
                f.write(ele + ",")
            f.write("\n")
            
            for line in most_fit_folders:
                f.write(f"{str(line[0])},{str(line[1])},\n")
            f.write("\n")
            
        with open("experiments/" + str(parameters['experiment_number']) + "/cumulative_runtime_stats_" + cumulative['unique_stamp'] + ".csv", 'w') as f:
            header = ("elapsed time","time per generation","time per network",)
            for ele in header:
                f.write(ele + ",")
            f.write("\n")
            f.write(f"{cumulative['elapsed']},{cumulative['elapsed'] / cumulative['generations']},{cumulative['elapsed'] / cumulative['networks']},\n")
            f.write("\n")