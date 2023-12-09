import threading
import pandas as pd
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from fuzzywuzzy import process

app = Flask(__name__)

# Load data from CSV file
data = pd.read_csv('C:/Users/jyoth/Downloads/data.csv')

# Load pre-trained GPT model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_response(user_input):
    # Tokenize the user's input
    user_tokens = set(tokenizer.tokenize(user_input.lower()))

    # Match user input with questions in the data.csv file
    matches = process.extractBests(user_input, data['question'], score_cutoff=90)

    if matches:
        # If a match is found, get the corresponding answer from the data
        answer = data.loc[matches[0][2], 'answer']
    else:
        # If not found in the data, generate a response using OpenAI GPT model
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        output = model.generate(input_ids,do_sample=True, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50,
                                top_p=0.95, temperature=0.7)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    response = get_response(user_input)
    return jsonify({'response': response})

def run_flask_app():
    app.run(host='0.0.0.0', port=5000)

# Start Flask app in a separate thread
threading.Thread(target=run_flask_app).start()
