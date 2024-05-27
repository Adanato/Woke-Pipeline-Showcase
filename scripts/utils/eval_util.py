import json
from tqdm import tqdm
import time
import os
import openai
from openai import ChatCompletion
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def load_prompt_format(filename, name):
    prompt_template = None
    output_format = None
    with open(filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data.get('name') == name:
                prompt_template = data.get('prompt_template')
                output_format = data.get('output_format')
                break  # Assuming only one entry per name, exit loop after finding the match
    return prompt_template, output_format


def extract_content(tag, text):
    # Find the starting position of the tag
    start_idx = text.find(tag)

    # If tag is not found, return None
    if start_idx == -1:
        return None

    # Extract the content after the tag
    content_after_tag = text[start_idx+len(tag):].strip()
    end_idx = text.find("#", start_idx + 1)
    return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()


def judger(judge, QApairs, prompt_template, output_format):
    '''
        Only supports openai models right now
    '''
    cnt = 0
    responses = []
    for QApair in tqdm(QApairs):

        while True:

            try:
                response = ChatCompletion.create(
                    model=judge,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template % QApair
                        }
                    ],
                    temperature=0,
                    max_tokens=10,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                responses.append(response)
                break

            except Exception as err:
                print('Exception occurs when calling GPT-4 for judge:', err)
                print('Will sleep for ten seconds before retry...')
                time.sleep(10)

    contents = [response.choices[0].message.content for response in responses]
    scores = [extract_content(output_format, content) for content in contents]
    return contents, scores

