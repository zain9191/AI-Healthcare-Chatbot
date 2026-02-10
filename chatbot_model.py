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
        try:
             regex = re.compile(pattern)
        except:
             return []
        return [item for item in self.cols if regex.search(item)]

    def sec_predict(self, symptoms_exp):
        df = pd.read_csv('Data/Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            if item in symptoms_dict:
                input_vector[symptoms_dict[item]] = 1
        
        return rf_clf.predict([input_vector])

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
        # Calculate condition (severity)
        calc_condition_msg = ""
        sum_severity = 0
        for item in symptoms_exp:
            if item in self.severityDictionary:
                sum_severity += self.severityDictionary[item]
        
        if (sum_severity * days) / (len(symptoms_exp) + 1) > 13:
            calc_condition_msg = "You should take the consultation from doctor."
            severity_message = "Severity: High"
        else:
            calc_condition_msg = "It might not be that bad but you should take precautions."
            severity_message = "Severity: Normal"

        second_prediction = self.sec_predict(symptoms_exp)
        
        result = {}
        result['severity_message'] = f"{severity_message}. {calc_condition_msg}"
        
        primary_desc = self.description_list.get(initial_disease, "No description available")
        secondary_desc = self.description_list.get(second_prediction[0], "No description available")
        
        if initial_disease == second_prediction[0]:
            result['text'] = f"You may have {initial_disease}"
            result['description_present'] = primary_desc
            result['description_second'] = primary_desc
        else:
            result['text'] = f"You may have {initial_disease} or {second_prediction[0]}"
            result['description_present'] = primary_desc
            result['description_second'] = secondary_desc

        result['precautions'] = self.precautionDictionary.get(initial_disease, [])
        return result
