import subprocess

# List of filenames
filenames = ["checkql.txt", "checkql_dirichlet.txt", "checkql_jm.txt","checkbm.txt","checkvsm.txt","checklsa.txt"]

# Open the results file in write mode
with open("results.txt", "w") as results_file:
    # Iterate over each filename
    for filename in filenames:
        # Define the command as a list of strings
        command = ["trec_eval", "-m", "map", "-m", "P.5", "-m", "ndcg", "cranqrel.trec.txt", filename]

        # Execute the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and capture output
        stdout, stderr = process.communicate()

        # Decode the output if needed
        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")

        # Write the output to the results file
        results_file.write(f"Output for {filename}:\n")
        results_file.write(stdout_str)
        results_file.write(stderr_str)
        results_file.write("\n")  # Add a newline for readability

        # Print the output to the console
        print(f"Output for {filename}:")
        print(stdout_str)
        print(stderr_str)