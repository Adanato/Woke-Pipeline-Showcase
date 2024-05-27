import csv
import json
import pandas as pd

def find_file_processor(file_path):
    file_path = str.lower(file_path)
    if "wokeytalkey" in file_path:
        print("WOKEY")
        if "hex_phi" in file_path:
            return wokey_hex_phi_file_processor
        elif "adv_bench" in file_path:
            return wokey_adv_bench_file_processor
    elif "hex-phi" in file_path:
        return hex_phi_file_processor
    elif "adv-bench" in file_path:
        print("ADV-Bench")
        return adv_bench_processor
    else:
        raise ValueError("{file_name} is not a known name")


def hex_phi_file_processor(file_path):
    """
    Loads prompts from a CSV file, where each prompt is in a new line.
    Assumes that the CSV file has no header and each line contains one prompt.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    list[str]: A list containing all prompts as strings.
    """
    prompts = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        # Create a reader object which will iterate over lines in the given csvfile
        reader = csv.reader(csvfile)
        for row in reader:
            if row:  # checking if the row is not empty
                prompts.append(row[0])  # assuming each row contains one prompt

    return prompts

def adv_bench_processor(file_path):
    """
    Loads prompts from a CSV file using pandas, where each prompt is in the 'goal' column.
    Assumes that the CSV file has headers including 'goal' and possibly others like 'target'.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    list[str]: A list containing all prompts from the 'goal' column as strings.
    """
    # Load the CSV file into a DataFrame, ensuring it reads the first line as headers
    df = pd.read_csv(file_path)

    # Check if 'goal' column exists in the DataFrame
    if 'goal' in df.columns:
        # Convert the 'goal' column to a list of strings
        prompts = df['goal'].tolist()
    else:
        # If 'goal' column is not present, return an empty list
        prompts = []

    return prompts
def wokey_adv_bench_file_processor(file_path):
    # Load the CSV file into a DataFrame, ensuring it reads the first line as headers
    df = pd.read_csv(file_path)
    
    # Check if 'prompt' and 'keep' columns exist in the DataFrame
    if 'prompt' in df.columns and 'keep' in df.columns:
        # Filter the DataFrame to keep only the rows where 'keep' is 1
        filtered_df = df[df['keep'] == 1]
        
        # Convert the 'prompt' column of the filtered DataFrame to a list of strings
        prompts = filtered_df['prompt'].tolist()
    else:
        # If 'prompt' or 'keep' column is not present, return an empty list
        prompts = []
    
    return prompts

def wokey_hex_phi_file_processor(file_path):
    # Load the CSV file into a DataFrame, ensuring it reads the first line as headers
    df = pd.read_csv(file_path)
    
    # Check if 'prompt' and 'keep' columns exist in the DataFrame
    if 'prompt' in df.columns and 'keep' in df.columns:
        # Filter the DataFrame to keep only the rows where 'keep' is 1
        filtered_df = df[df['keep'] == 1]
        
        # Convert the 'prompt' column of the filtered DataFrame to a list of strings
        prompts = filtered_df['prompt'].tolist()
    else:
        # If 'prompt' or 'keep' column is not present, return an empty list
        prompts = []
    
    return prompts

def load_woke_template(file_path="../configs/woke_templates.json", name="woke-template-v1"):

    with open(file_path, "r") as file:
        templates = json.load(file)

    return templates[name]
