from chatbot_model import HealthcareChatbot
import pyttsx3

def readn(nstr):
    try:
        engine = pyttsx3.init()
        engine.setProperty('voice', "english+f5")
        engine.setProperty('rate', 130)
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()
    except:
        pass

if __name__ == "__main__":
    bot = HealthcareChatbot()
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t",end="->")
    name = input("")
    print("Hello, ", name)

    while True:
        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        symptom = input("")
        matches = bot.check_pattern(symptom)
        
        if len(matches) > 0:
            print("searches related to input: ")
            for num, it in enumerate(matches):
                print(num, ")", it)
            
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                try:
                    conf_inp = int(input(""))
                    symptom = matches[conf_inp]
                except:
                    print("Invalid input, using first match")
                    symptom = matches[0]
            else:
                symptom = matches[0]
                
            # Now we have the confirmed symptom
            # Get the tree info
            tree_info = bot.get_symptom_tree_info(symptom)
            present_disease = tree_info['predicted_disease']
            related_symptoms = tree_info['related_symptoms']
            
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in related_symptoms:
                if syms == symptom:
                    symptoms_exp.append(syms)
                    continue
                    
                print(syms,"? : ",end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if inp == "yes":
                    symptoms_exp.append(syms)
            
            try:
                days = int(input("Okay. From how many days ? : "))
            except:
                days = 1
                
            result = bot.final_prediction(present_disease, symptoms_exp, days)
            
            print(result['text'])
            print(result['description_present'])
            if 'description_second' in result and result['description_second'] != result['description_present']:
                print(result['description_second'])
            
            print("Take following measures : ")
            for i, j in enumerate(result['precautions']):
                print(i+1, ")", j)
                
            print(result['severity_message'])
            break
        else:
             print("Enter valid symptom.")

