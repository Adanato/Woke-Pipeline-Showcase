from utils.config_util import load_dataset_category_dict
from utils.file_util import load_woke_template
import argparse
from openai import OpenAI
import time
import json
from dotenv import load_dotenv
import os
load_dotenv()


def process_json_files(input_dir, woke_template_file_name, dataset_file_path):
    requests = []
    file_names = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    template = load_woke_template()
    category = load_dataset_category_dict(dataset_file_path)
    for filename in file_names:
        woke_model_name = filename.split('_')[0]

        with open(f"{input_dir}/{filename}", 'r') as file:
            # Load the data from the file
            prompts = json.load(file)

        for tag, prompt_object in enumerate(prompts):
            # Initialize an empty list to hold all the request objects
            prompt = prompt_object['question']
            # Define the values for the Python object
            custom_id = f"{woke_model_name}_{tag}"
            method = "POST"
            url = "/v1/chat/completions"
            model = "gpt-4-turbo"
            content = template % (prompt, category)
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
                    "temperature": 0.7, "max_tokens": 500, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0
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


def retrieve_batch(client, batch):
    return client.batches.retrieve(batch.id)


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
    parser.add_argument('--woke_template_file_name', type=str, default='../configs/woke_templates.json',
                        help='Path to the judge prompt file.')
    parser.add_argument('--output_dir', type=str, default='./0_bash_files/0_batch_data/',
                        help='Path and name of the output file to save the requests.')
    parser.add_argument('--batch_name', type=str, default='batch_output',
                        help='Name of the batch output file.')
    parser.add_argument('--dataset_file_path', type=str, default='batch_output',
                        help='Name of dataset path')
    args = parser.parse_args()

    input_dir = args.input_dir
    woke_template_file_name = args.woke_template_file_name
    output_dir = args.output_dir
    batch_name = args.batch_name
    dataset_file_path = args.dataset_file_path
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "API Key not found. Please set the OPENAI_API_KEY environment variable.")

    requests = process_json_files(
        input_dir=input_dir, woke_template_file_name=woke_template_file_name, dataset_file_path=dataset_file_path)
    save_requests_to_file(requests, f'{output_dir}/batch_request.jsonl')

    if not confirm_action("Do you want to proceed with sending the request to OpenAI?"):
        print("Request canceled.")
        exit(0)

    client, response = upload_file_to_openai(
        f'{output_dir}/batch_request.jsonl', api_key)
    batch = create_batch(client, response)

    while True:
        try:
            retrieved_batch = retrieve_batch(client, batch)
            if retrieved_batch.status == 'completed':
                break
            else:
                raise Exception(
                    f"Batch processing retrieval failed with status: {retrieved_batch.status}")
        except Exception as e:
            print(f"Error retrieving batch: {e}")
            print("Retrying in 30 seconds...")
            time.sleep(30)

    text_content = get_file_content(client, retrieved_batch)

    if text_content:
        parsed_objects = parse_multiple_json_objects(text_content)
        save_as_jsonl(parsed_objects, f"{output_dir}/batch_raw_outputs.jsonl")
