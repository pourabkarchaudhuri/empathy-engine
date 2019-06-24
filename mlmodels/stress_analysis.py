# Stress Levels:
# 90 to 100 : Extreme/Angry
# 80 to 90 : Critical Shocks/Unresponsive
# 70 to 80 : Worried
# 50 to 70 : Slighly responsive
# Below 50 : Responsive

import random
intent_state = False
intent_state_name = False
global_intent_repeat = 0
is_completed = False

def stress_analyzer(sentiment, intent, stress):
    trigger = None
    is_responsive = True
    global intent_state
    global intent_state_name
    global is_completed
    if intent_state is not True:
        print("Fresh Start")
    else:
        if intent_state_name == intent:
            print("Repeating Intent : {}".format(intent))
            if(stress>90):
                stress = stress + random.randint(1, 5)
                trigger = 'dont_annoy'
            elif stress>80 and stress<=90:
                trigger = None
                is_responsive = False
            elif stress>70 and stress<=80 :
                trigger = 'worry'
            elif stress>0 and stress<=70:
                trigger = None
        else:
            print("New Intent : {}".format(intent))

    reaction = ""
    
    print("Stress Input : {}".format(stress))
    if stress<100:
        if sentiment=='negative':
            stress = stress + random.randint(1, 5)
        elif sentiment=='positive':
            stress = stress - random.randint(1, 5)
        else:
            print("No change in stress")
    else:
        print("Full Stress")
    if stress>=100:
        stress=100
    
    if stress>90:
        reaction = "extreme"
    elif stress>80 and stress<=90:
        reaction = "shock"
    elif stress>70 and stress<=80:
        reaction = "worried"
    elif stress>60 and stress<=70:
        reaction = "responsive"
    elif stress<=60:
        reaction = "relieved"
        is_completed = True

    print("Current Stress Level {}, with Reaction {}".format(stress, reaction))
    intent_state_name = intent
    intent_state = True
    print("Intent State Name is {}, and State {}".format(intent_state_name, intent_state))
    response = {
        'reaction':reaction,
        'stress': stress,
        'trigger': trigger,
        'responsive': is_responsive,
        'completion': is_completed
    }
    print("Output Payload : ", response)
    return response

# stress_analyzer('negative', 'greetings', 78)
    