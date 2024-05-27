import time
import json
import os
from utils.eval_util import load_prompt_format
from openai import OpenAI
import argparse
from utils.progress_util import progress_bar

def process_json_files(directory_path, judge_prompt_filename):
    requests = []

    file_names = [f for f in os.listdir(directory_path) if f.endswith('.json')]

    for filename in file_names:
        json_objects = []
        
        with open(os.path.join(directory_path, filename), 'r') as file:
            for line in file:
                try:
                    json_object = json.loads(line.strip())  # Strip newline characters
                    json_objects.append(json_object)

                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON from line: {line}, error: {e}")
        
        model_name = filename.split('_')[0]
        for i, object in enumerate(json_objects):
            # Initialize an empty list to hold all the request objects
            answer = object['generated']
            
            raw_prompt = object['raw_prompt']
            QA_Pair = (raw_prompt, answer)
            woke_id = object['woke_id']
            woke_model = object['woke_model']
            # Define the values for the Python object
            custom_id = f"{model_name}_{i}_woketag_{woke_model}_{woke_id}"
            method = "POST"
            url = "/v1/chat/completions"
            model = "gpt-4-turbo"
            prompt_template,outputformat = load_prompt_format(judge_prompt_filename,'base-general-v3')

            content = prompt_template % QA_Pair

            messages = [
                {"role": "user", "content": content}
            ]

            # Create a dictionary that represents the request
            request = {
                "custom_id": custom_id,
                "method": method,
                "url": url,
                "body": {
                    "model": model,
                    "messages": messages,
                    "temperature": 0, "max_tokens": 128, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0
                }
            }

            # Add the dictionary to the list
            requests.append(request)
    return requests

def save_requests_to_file(requests, output_file_name):
    with open(output_file_name, 'w') as f:
        for request in requests:
            json_string = json.dumps(request)
            f.write(json_string + '\n')


def upload_file_to_openai(file_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(file_path, "rb") as file:
        response = client.files.create(file=file, purpose="batch")
    return client, response


def create_batch(client, response):
    batch = client.batches.create(
        input_file_id=response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch


def retrieve_batch(client, batch_id):
    return client.batches.retrieve(batch_id)


def get_file_content(client, batch):
    response = client.files.content(batch.output_file_id)
    if hasattr(response, 'content'):
        text_content = response.content.decode('utf-8')
        return text_content
    return None


def parse_multiple_json_objects(json_string):
    objects = []
    remaining_string = json_string.strip()
    while remaining_string:
        try:
            obj, idx = json.JSONDecoder().raw_decode(remaining_string)
            objects.append(obj)
            remaining_string = remaining_string[idx:].strip()
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            break
    return objects


def save_as_jsonl(objects, filename):
    with open(filename, 'a') as file:
        for obj in objects:
            json_line = json.dumps(obj)
            file.write(json_line + '\n')


def confirm_action(message):
    while True:
        user_input = input(f"{message} (y/n): ")
        if user_input.lower() == 'y':
            return True
        elif user_input.lower() == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process JSON files and create OpenAI batch requests.')
    parser.add_argument('--input_dir', type=str, default='./0_bash_files/0_outputs/more_woke_model_900_outputs/',
                        help='Path to the directory containing JSON files to process.')
    parser.add_argument('--judge_prompt_filename', type=str, default='./judge_prompt.jsonl',
                        help='Path to the judge prompt file.')
    parser.add_argument('--output_dir', type=str, default='./0_bash_files/0_batch_data/',
                        help='Path and name of the output file to save the requests.')
    parser.add_argument('--batch_name', type=str, default='batch_output',
                        help='Name of the batch output file.')
    args = parser.parse_args()

    input_dir = args.input_dir
    judge_prompt_filename = args.judge_prompt_filename
    output_dir = args.output_dir
    batch_name = args.batch_name

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API Key not found. Please set the OPENAI_API_KEY environment variable.")

    batch_ids_file = f"{output_dir}/batch_ids.jsonl"

    if os.path.exists(batch_ids_file):
        with open(batch_ids_file, "r") as f:
            batch_ids = [json.loads(line) for line in f]
        if batch_ids:
            print("Available batch IDs:")
            for i, batch_id in enumerate(batch_ids, start=1):
                print(f"{i}. {batch_id['batch_id']}")
            while True:
                batch_index = input("Enter the number of the batch ID you want to retrieve (or press Enter to create a new batch): ")
                if batch_index == "":
                    break
                try:
                    batch_index = int(batch_index) - 1
                    if 0 <= batch_index < len(batch_ids):
                        selected_batch_id = batch_ids[batch_index]["batch_id"]
                        client = OpenAI(api_key=api_key)
                        while True:
                            try:
                                retrieved_batch = retrieve_batch(client, selected_batch_id)
                                if retrieved_batch.status == 'completed':
                                    break
                                else:
                                    raise Exception(
                                        f"Batch processing failed with status: {retrieved_batch.status}")
                            except Exception as e:
                                print(f"Error retrieving batch: {e}")
                                print("Retrying in 30 seconds...\n")
                                progress_bar(30)
                        text_content = get_file_content(client, retrieved_batch)
                        if text_content:
                            parsed_objects = parse_multiple_json_objects(text_content)
                            save_as_jsonl(parsed_objects, f"{output_dir}/batch_raw_outputs.jsonl")
                        exit(0)
                    else:
                        print("Invalid batch number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

    requests = process_json_files(input_dir, judge_prompt_filename)
    save_requests_to_file(requests, f'{output_dir}/batch_request.jsonl')

    if not confirm_action("Do you want to proceed with sending the request to OpenAI?"):
        print("Request canceled.")
        exit(0)

    client, response = upload_file_to_openai(
        f'{output_dir}/batch_request.jsonl', api_key)
    batch = create_batch(client, response)

    with open(batch_ids_file, "a") as f:
        f.write(json.dumps({"batch_id": batch.id}) + "\n")

    while True:
        try:
            retrieved_batch = retrieve_batch(client, batch.id)
            if retrieved_batch.status == 'completed':
                break
            else:
                raise Exception(
                    f"Batch processing failed with status: {retrieved_batch.status}")
        except Exception as e:
            print(f"Error retrieving batch: {e}")
            print("Retrying in 30 seconds...\n")
            progress_bar(30)

    text_content = get_file_content(client, retrieved_batch)
    if text_content:
        parsed_objects = parse_multiple_json_objects(text_content)
        save_as_jsonl(parsed_objects, f"{output_dir}/batch_raw_outputs.jsonl")