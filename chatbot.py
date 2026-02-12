import re
import numpy as np
import joblib
import pandas as pd
from sklearn.tree import _tree
from utils import getDescription, getSeverityDict, getprecautionDict, getspecialistDict

# Global variables
clf = None
le = None
training_cols = None
symptoms_dict = {}
description_list = {}
severityDictionary = {}
precautionDictionary = {}
specialistDictionary = {}
reduced_data = None

def load_resources():
    global clf, le, training_cols, symptoms_dict, reduced_data
    global description_list, severityDictionary, precautionDictionary, specialistDictionary
    
    try:
        clf = joblib.load('saved_model/decision_tree.joblib')
        le = joblib.load('saved_model/label_encoder.joblib')
        training_cols = joblib.load('saved_model/training_cols.joblib')
    except Exception as e:
        print("CRITICAL: Models not found. Run 'python model.py' first.")
        return

    symptoms_dict = {symptom: index for index, symptom in enumerate(training_cols)}
    
    description_list = getDescription()
    severityDictionary = getSeverityDict()
    precautionDictionary = getprecautionDict()
    specialistDictionary = getspecialistDict()
    
    try:
        training = pd.read_csv('Data/Training.csv')
        reduced_data = training.groupby(training['prognosis']).max()
    except:
        print("Warning: Data/Training.csv not found. Prediction details might fail.")

# Initialize immediately
load_resources()

def check_pattern(dis_list, inp):
    # YOUR CODE: Handles sentences like "I have fatigue"
    if not inp:
        return 0, []
    
    inp = inp.lower().replace('_', ' ') 
    found_symptoms = []
    
    for symptom in dis_list:
        symptom_readable = symptom.replace('_', ' ')
        if symptom_readable in inp:
            found_symptoms.append(symptom)
            
    if len(found_symptoms) > 0:
        return 1, found_symptoms
    else:
        return 0, []

def sec_predict(symptoms_exp):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    
    input_df = pd.DataFrame([input_vector], columns=training_cols)
    predicted_encoded = clf.predict(input_df)
    return le.inverse_transform(predicted_encoded)[0]

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def calc_condition(exp, days):
    sum_severity = 0
    for item in exp:
        if item in severityDictionary:
            sum_severity += severityDictionary[item]
    
    if (sum_severity * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from a doctor immediately."
    else:
        return "It might not be that bad, but you should take precautions."

def predict_initial(symptom):
    if symptom not in symptoms_dict:
        return ["Unknown"], []

    tree_ = clf.tree_
    feature_name = [
        training_cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    symptoms_present = []

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if name == symptom:
                val = 1
            else:
                val = 0
            if val <= threshold:
                return recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            return present_disease, list(symptoms_given)

    return recurse(0, 1)

def predict_final(initial_disease, symptoms_exp, days):
    second_prediction = sec_predict(symptoms_exp)
    severity_message = calc_condition(symptoms_exp, days)
    
    text = ""
    if initial_disease == second_prediction:
        text = f"{initial_disease}"
    else:
        text = f"{initial_disease} or {second_prediction}"

    description_present = description_list.get(initial_disease, "No description available.")
    description_second = description_list.get(second_prediction, "No description available.")
    
    precautions = precautionDictionary.get(initial_disease, [])
    specialist_present = specialistDictionary.get(initial_disease, "General Physician")
    specialist_second = specialistDictionary.get(second_prediction, "General Physician")

    return (text, description_present, description_second, precautions, severity_message, specialist_present, specialist_second)