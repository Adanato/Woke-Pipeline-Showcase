from flask import Flask, request, render_template, session, redirect, url_for
import csv
import json
import time
import copy
import os
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # This key is required for session management

# Load the questions and answers
questions_file = 'batch_1.json'

questions = {}

with open(questions_file, 'r') as jsonlfile:
    data = json.load(jsonlfile)
    for i, object in enumerate(data):
        questions[i] = {
            'prompt': object['prompt'],
            'dataset': object['dataset']
        }

question_ids = []
answers_by_question = {}

with open(questions_file, 'r') as jsonlfile:
    data = json.load(jsonlfile)
    for i, object in enumerate(data):
        question_id = i
        question_ids.append(i)
        answers_by_question[question_id] = []
        answers_by_question[question_id].append(object['models'])
question_ids.sort()
print("Question IDs to be evaluated:", question_ids)

@app.route('/', methods=['GET'])
def index():
    log_file_path = 'human_eval.json'
    existing_data = []

    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r') as logfile:
                try:
                    existing_data = json.load(logfile)
                except json.JSONDecodeError:
                    print("Error: Invalid JSON data in the log file.")
                    existing_data = []
        except FileNotFoundError:
            existing_data = []

    last_question_index = len(existing_data)

    if last_question_index >= len(question_ids):
        return "Thank you for completing the evaluation!"

    session['question_index'] = last_question_index

    question_id = question_ids[session['question_index']]
    return redirect(url_for('evaluate', question_id=question_id))


@app.route('/question/<int:question_id>', methods=['GET', 'POST'])
def evaluate(question_id):
    if request.method == 'POST':
        data = {
            'scores': [],
            'prompt': {
                'dataset': questions[question_id]['dataset'],
                'text': questions[question_id]['prompt'],
                'unreasonable': 0,
                'harmful': 0
            }
        }

        # Collect scores from form inputs
        for key, value in request.form.items():
            if key.startswith('score_'):
                answer_index = int(key.split('_')[1]) - 1
                answer = answers_by_question[question_id][0][answer_index]
                gen_model = answer['generating_model']
                woke_model = answer['woke_model']
                gpt_score = answer['score']
                score = {
                    'generating_model': gen_model,
                    'woke_model':woke_model ,
                    'model_response': answer['generated'].replace("<br>", "\n"),
                    'human_score': float(value),
                    'gpt_judge_score': gpt_score,
                }
                data['scores'].append(score)

        # Handle prompt evaluation
        data['prompt']['unreasonable'] = int(request.form.get('prompt_unreasonable', 0))
        data['prompt']['harmful'] = int(request.form.get('prompt_harmful', 0))

        # Read the existing data, update it, and write it back
        file_path = 'human_eval.json'
        try:
            with open(file_path, 'r') as jsonfile:
                # Load existing data into a list
                try:
                    existing_data = json.load(jsonfile)
                except json.JSONDecodeError:
                    existing_data = []
        except FileNotFoundError:
            existing_data = []

        # Append new data
        existing_data.append(data)

        # Save the updated data to a JSON file
        with open(file_path, 'w') as jsonfile:
            json.dump(existing_data, jsonfile, indent=4)

        return redirect(url_for('index'))

    if question_id > question_ids[-1]:
        return "Thank you for completing the evaluation!"

    question = questions[question_id]['prompt']
    question = question.replace("\n", "<br>")

    answers = answers_by_question.get(question_id, [])[0]
    # change all "\n" to "<br>" for better display
    for answer in answers:
        answer['generated'] = answer['generated'].replace("\n", "<br>")

    # Find location of the current question in the list of question_ids
    cur = question_ids.index(question_id)
    total = len(question_ids)

    # Pass the current question and its answers to the template
    return render_template('evaluation.html', question_id=question_id, question=question, answers=answers, cur=cur, total=total)

if __name__ == '__main__':
    app.run(debug=True)