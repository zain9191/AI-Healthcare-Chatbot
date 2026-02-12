from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import check_pattern, training_cols, predict_initial, predict_final

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_pattern", methods=["POST"])
def handle_check_pattern():
    pattern = request.json.get("pattern")
    conf, matches = check_pattern(training_cols, pattern)
    return jsonify({"matches": matches})

@app.route("/predict_initial", methods=["POST"])
def handle_predict_initial():
    symptom = request.json.get("symptom")
    try:
        predicted_disease, related_symptoms = predict_initial(symptom)
        return jsonify({
            "predicted_disease": predicted_disease[0],
            "related_symptoms": related_symptoms
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/predict_final", methods=["POST"])
def handle_predict_final():
    data = request.json
    initial_disease = data.get("initial_disease")
    symptoms_exp = data.get("symptoms_exp")
    days = int(data.get("days"))
    
    try:
        text, desc1, desc2, precautions, severity, specialist1, specialist2 = predict_final(
            initial_disease, symptoms_exp, days
        )
        return jsonify({
            "text": text,
            "description_present": desc1,
            "description_second": desc2,
            "precautions": precautions,
            "severity_message": severity,
            "specialist_present": specialist1,
            "specialist_second": specialist2,
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)