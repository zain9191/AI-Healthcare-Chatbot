import csv
import os

def getSeverityDict():
    severityDictionary = dict()
    # Matches your screenshot: Capital 'S'
    filename = 'MasterData/Symptom_severity.csv'
    
    try:
        with open(filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2 and row[1].strip().isdigit():
                    _diction = {row[0].strip(): int(row[1].strip())}
                    severityDictionary.update(_diction)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
    return severityDictionary

def getDescription():
    description_list = dict()
    # Matches your screenshot: Capital 'D'
    filename = 'MasterData/symptom_Description.csv'
    
    try:
        with open(filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:
                    _description = {row[0].strip(): row[1].strip()}
                    description_list.update(_description)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
    return description_list

def getprecautionDict():
    precautionDictionary = dict()
    filename = 'MasterData/symptom_precaution.csv'
    
    try:
        with open(filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 5: 
                    _prec = {row[0].strip(): [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]}
                    precautionDictionary.update(_prec)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
    return precautionDictionary

def getspecialistDict():
    specialistDictionary = dict()
    filename = 'MasterData/symptom_specialist.csv'
    
    try:
        with open(filename, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) >= 2:
                    if row[0].strip().lower() != "disease":
                        specialistDictionary[row[0].strip()] = row[1].strip()
    except FileNotFoundError:
        pass 
    return specialistDictionary