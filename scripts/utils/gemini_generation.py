import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY=os.environ["GEMINI_API_KEY"]


generation_config = {
  "temperature": 0,
  #"max_output_tokens": 100,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]
genai.configure(api_key=GEMINI_API_KEY)

import time

def get_response_gemini(prompt, model_name):
    model = genai.GenerativeModel(
        model_name=model_name,
        safety_settings=safety_settings,
        generation_config=generation_config,
    )
    print("hello")
    response = None
    try:
        response = model.generate_content(prompt)
        if model_name == "gemini-1.5-pro":
            time.sleep(32)  # Free version limiter. Change after may 2 if on a paid plan
        elif model_name == "gemini-1.5-flash":
            time.sleep(int(60/14))
        
        print(response.text)
        return response.text
    except Exception as e:
        print("An error occurred:", str(e))
        # Handle the exception or provide feedback based on the prompt
        if response.prompt_feedback:
          return str(response.prompt_feedback)
        
        raise e
    
    
    