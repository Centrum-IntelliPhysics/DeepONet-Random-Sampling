import os

# Define the values for ntrain and neval
ntrain_values = [8000, 4000, 2000, 1000, 500]
neval_values = [1, 10, 50, 100]
seed_values = [0, 1, 2, 3 ,4, 5, 6, 7, 8, 9]

# Iterate over each combination of ntrain, neval and seed
for ntrain in ntrain_values:
    for neval in neval_values:
        for seed in seed_values:
            
            # Construct the command to run the script with the current values of ntrain, neval and seed
            command = f"python DeepONet_analysis.py -ntrain {ntrain} -neval {neval} -seed {seed}"
        
            # Execute the command
            os.system(command)
