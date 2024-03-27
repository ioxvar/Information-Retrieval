import os
from dotenv import load_dotenv
import subprocess

load_dotenv()
filenames = [os.getenv("QL_OUTPUT"), os.getenv("QL_D_OUTPUT"),os.getenv("QL_JM_OUTPUT"),os.getenv("BMS_OUTPUT"),os.getenv("VSM_OUTPUT"),os.getenv("LSA_OUTPUT")]
display_names = ['Query Likelihood Model','Query Likelihood Model - Dirichlet Smoothing','Query Likelihood Model - JM Smoothing','BM25 model','Vector Space Model','Latent Semantic Analysis']

with open(os.getenv("RESULTS"), "w") as results_file:
    for filename,display_name in zip(filenames,display_names):
        try:
            terminal_command = ["trec_eval", "-m", "map", "-m", "P.5", "-m", "ndcg", "cranqrel.trec.txt", filename]
            process = subprocess.Popen(terminal_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")

            results_file.write(f"Output for {display_name}:\n")
            results_file.write(stdout_str)
            results_file.write(stderr_str)
            results_file.write("\n") 

            print(f"Output for {display_name}:")
            print(stdout_str)
            print(stderr_str)
        except Exception as e:
            print(f"An error occurred while running subprocess: {e}")
