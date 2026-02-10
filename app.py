from flask import Flask, render_template, request, jsonify
from chatbot_model import HealthcareChatbot
import traceback

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize the chatbot model
try:
    bot = HealthcareChatbot()
    print("Chatbot model initialized successfully.")
except Exception as e:
    print(f"Error initializing chatbot model: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': bot.get_all_symptoms()})

@app.route('/check_pattern', methods=['POST'])
def check_pattern():
    data = request.get_json()
    pattern = data.get('pattern', '')
    matches = bot.check_pattern(pattern)
    return jsonify({'matches': matches})

@app.route('/predict_initial', methods=['POST'])
def predict_initial():
    data = request.get_json()
    symptom = data.get('symptom')
    if not symptom:
        return jsonify({'error': 'Symptom is required'}), 400
    
    try:
        result = bot.get_symptom_tree_info(symptom)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_final', methods=['POST'])
def predict_final():
    data = request.get_json()
    initial_disease = data.get('initial_disease')
    symptoms_exp = data.get('symptoms_exp') # List of confirmed symptoms
    days = data.get('days', 0)
    
    if not initial_disease or not symptoms_exp:
        return jsonify({'error': 'Initial disease and symptoms are required'}), 400
        
    try:
        # User entered days as integer or string
        try:
            days = int(days)
        except:
            days = 1
            
        result = bot.final_prediction(initial_disease, symptoms_exp, days)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
