import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)

class HealthcareChatbot:
    def __init__(self):
        self.training = pd.read_csv('Data/Training.csv')
        self.testing = pd.read_csv('Data/Testing.csv')
        self.cols = self.training.columns[:-1]
        self.x = self.training[self.cols]
        self.y = self.training['prognosis']
        self.reduced_data = self.training.groupby(self.training['prognosis']).max()

        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.y)
        self.y_encoded = self.le.transform(self.y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y_encoded, test_size=0.33, random_state=42)

        self.clf = DecisionTreeClassifier()
        self.clf.fit(self.x_train, self.y_train)
        
        # Initialize and train the secondary model (Random Forest) once
        # Using RandomForest for better probability estimation than a single Decision Tree
        from sklearn.ensemble import RandomForestClassifier
        self.rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_clf.fit(self.x, self.y_encoded)

        # Secondary model (optional/legacy)
        self.model = SVC()
        self.model.fit(self.x_train, self.y_train)

        self.severityDictionary = dict()
        self.description_list = dict()
        self.precautionDictionary = dict()
        self.symptoms_dict = {}

        for index, symptom in enumerate(self.x):
            self.symptoms_dict[symptom] = index

        self._get_severity_dict()
        self._get_description()
        self._get_precaution_dict()

    def _get_severity_dict(self):
        try:
            with open('MasterData/Symptom_severity.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if len(row) >= 2:
                        self.severityDictionary[row[0]] = int(row[1])
        except Exception as e:
            print(f"Error loading severity: {e}")

    def _get_description(self):
        try:
            with open('MasterData/symptom_Description.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if len(row) >= 2:
                        self.description_list[row[0]] = row[1]
        except Exception as e:
            print(f"Error loading description: {e}")

    def _get_precaution_dict(self):
        try:
            with open('MasterData/symptom_precaution.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if len(row) >= 5:
                        self.precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
        except Exception as e:
            print(f"Error loading precautions: {e}")

    def get_all_symptoms(self):
        return list(self.cols)

    def check_pattern(self, pattern):
        pattern = pattern.replace(' ', '_')
        # Exact/Regex match first
        try:
             regex = re.compile(pattern, re.IGNORECASE)
             matches = [item for item in self.cols if regex.search(item)]
        except:
             matches = []
        
        # If no strict matches, try fuzzy matching
        if not matches:
             import difflib
             # Get close matches (cutoff=0.6 allows for some typos)
             fuzzy_matches = difflib.get_close_matches(pattern, self.cols, n=5, cutoff=0.6)
             matches.extend(fuzzy_matches)
             
        # Dedup just in case
        return sorted(list(set(matches)))

    def normalize_probabilities(self, probs):
        """Helper to format probabilities as percentages"""
        return [round(p * 100, 2) for p in probs]

    def predict_top_diseases(self, symptoms_exp, top_k=3):
        # Create input vector
        input_vector = np.zeros(len(self.symptoms_dict))
        for item in symptoms_exp:
            if item in self.symptoms_dict:
                input_vector[self.symptoms_dict[item]] = 1
        
        # Get probabilities
        probs = self.rf_clf.predict_proba([input_vector])[0]
        
        # Get top k indices
        top_indices = probs.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if probs[idx] > 0: # Only return if there's some probability
                disease = self.le.inverse_transform([idx])[0]
                probability = probs[idx] * 100
                description = self.description_list.get(disease)
                if not description or description.strip() == '' or description == 'undefined':
                    description = f"Aucune description trouvée pour {disease}. (Maladie prédite brute : {disease})"
                precautions = self.precautionDictionary.get(disease)
                if not precautions or all([not p or p == 'undefined' for p in precautions]):
                    precautions = [f"Aucune précaution trouvée pour {disease}. (Maladie prédite brute : {disease})"]
                results.append({
                    'disease': disease,
                    'probability': round(probability, 2),
                    'description': description,
                    'precautions': precautions,
                    'raw_disease': disease
                })
        
        return results

    def sec_predict(self, symptoms_exp):
        # Legacy support wrapper for the new model
        results = self.predict_top_diseases(symptoms_exp, top_k=1)
        if results:
            return [results[0]['disease']]
        return ["Unknown"]

    def print_disease(self, node):
        node = node[0]
        val  = node.nonzero() 
        disease = self.le.inverse_transform(val[0])
        return list(map(lambda x:x.strip(),list(disease)))

    def get_symptom_tree_info(self, symptom_name):
        tree_ = self.clf.tree_
        feature_name = [
            self.cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        # Determine the leaf node for the given symptom (forcing existence)
        node = 0
        while tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if name == symptom_name:
                val = 1
            else:
                val = 0
            if val <= threshold:
                node = tree_.children_left[node]
            else:
                node = tree_.children_right[node]
        present_disease = self.print_disease(tree_.value[node])
        # Get symptoms associated with this disease
        red_cols = self.reduced_data.columns
        symptoms_given = red_cols[self.reduced_data.loc[present_disease].values[0].nonzero()]
        return {
            'predicted_disease': present_disease[0],
            'related_symptoms': list(symptoms_given)
        }

    def final_prediction(self, initial_disease, symptoms_exp, days):
        # Calcul de la sévérité
        calc_condition_msg = ""
        sum_severity = 0
        for item in symptoms_exp:
            if item in self.severityDictionary:
                sum_severity += self.severityDictionary[item]
        if (sum_severity * days) / (len(symptoms_exp) + 1) > 13:
            calc_condition_msg = "Vous devriez consulter un médecin."
            severity_message = "Gravité : Élevée"
        else:
            calc_condition_msg = "Ce n'est peut-être pas grave mais prenez des précautions."
            severity_message = "Gravité : Normale"
        # Prédiction des maladies les plus probables
        top_predictions = self.predict_top_diseases(symptoms_exp, top_k=5)
        result = {}
        result['severity_message'] = f"{severity_message}. {calc_condition_msg}"
        if top_predictions:
            primary_pred = top_predictions[0]
            result['primary_disease'] = primary_pred['disease']
            result['primary_probability'] = primary_pred['probability']
            result['description_present'] = primary_pred['description']
            result['precautions'] = primary_pred['precautions']
            # Texte détaillé
            text = f"Sur la base de vos symptômes, le diagnostic le plus probable est <b>{primary_pred['disease']}</b> (Confiance : {primary_pred['probability']}%)"
            if len(top_predictions) > 1:
                text += "<br><br>Autres possibilités :<br>"
                for pred in top_predictions[1:]:
                    text += f"- <b>{pred['disease']}</b> : {pred['probability']}%<br>"
            result['text'] = text
            result['detailed_predictions'] = top_predictions
        else:
            result['text'] = "Impossible de faire une prédiction fiable avec ces symptômes."
            result['precautions'] = []
        return result
