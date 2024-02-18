# import population
import os
import shutil

class Reporting():
    
    # def simulation_parameters()
    
    def get_population_stats(population):
        # Get things like the number of networks
        pass
    
    def census(sim_population):
        # Gathers and returns the stats of a population
        results = []
        # population_size = population.Population.get_population_size(sim_population)
        population_size = sim_population.get_population_size()
        
        # each network's meta data, and fitness
        for i in range(population_size):
            # nn_meta = population.Population.get_neural_network_def(sim_population, i)
            nn_meta = sim_population.get_neural_network_def(i)
            nn_fitness = sim_population.get_nn_fitness(i)
            results.append([nn_meta, nn_fitness])
        return results
    
    
    def create_folders(parameters):
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
        #  already exists, to accommodate the subdirectories.
        try:
            os.mkdir("experiments/" + str(maindir))
            os.mkdir("experiments/" + str(maindir) + "/!-- " + str(comment))
        except:
            print(f"\n\n -----------------------------------------------------\n")
            print(f"Note, the main directory {maindir} already exists. " +
                "Subfolders are being mixed with new and previous data.")
            print(f"\n -----------------------------------------------------\n")

        # But subfolders must not be overwritten if they exist as this is where
        #  the data is stored.
        try:
            os.mkdir("experiments/" + str(maindir) + "/" + str(subdir))
        except:
            print(f"\n\nFATAL. The experiment directory {fulldir} already exists.\n\n")
            quit(1)
        
    
    def output_to_csv(parameters, report):
        maindir = parameters['experiment_number']
        subdir = parameters['subfolder']
        fulldir = maindir + "/" + subdir
        
        if subdir == 0 or maindir == 0:
            return
        # Assumes that the subdirectory exists as the create_folders function was called earlier.      
        
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
       
        # Write the details for each network in each generation
        with open("experiments/" + str(fulldir) + "/nn_and_results_data.csv", 'w') as f:
            header = ("generation","stage","#","serial_number","checksum","parent_1","parent_2","hidden_weights","output_weights","hidden_checksum","output_checksum","input","hidden_type","hidden_neurons","hidden_activation","output_type","output_count","output_activation","fitness")
            for ele in header:
                f.write(ele + ",")
            f.write("\n")
            
            # Write the initial population details
            for nn in range(len(report['initial_population'])):
                # Add filler for initial population
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


        # Write the output for each network in each generation
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

       
        # Write the summary for each generation
        
        # number of networks
        # maximum fitness
        # average fitness
        
        with open("experiments/" + str(fulldir) + "/generation_summary.csv", 'w') as f:
            header = ("generation","number_of_networks","maximum_fitness","average_fitness")
            for ele in header:
                f.write(ele + ",")
            f.write("\n")
            
        sim_avg_fitness = 0
        for gen in range(len(report['generations'])):
            fitnesses = []
            # Calculate and write the output for each network in each generation, after evaluation
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
