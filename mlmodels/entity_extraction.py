import spacy
nlp = spacy.load('en_core_web_sm')

from semantic.numbers import NumberService
from semantic.dates import DateService
number_service = NumberService()
date_service = DateService()
# print("Time now : ", date_service.extractTime("3 o clock"))

def named_entity_extraction(text):
    doc = nlp(text)

    entity = []
    for ent in doc.ents:
        # print("Iterating : ", ent)

        if ent.label_ == 'ORG':
            name = ent.text
            description = "organization"
        elif ent.label_ == 'GPE':
            name = ent.text
            description = "geopolitical area"
        elif ent.label_ == 'MONEY':
            name = ent.text
            description = "monetary value"
        elif ent.label_ == 'PERSON':
            name = ent.text
            description = "public figure"
        elif ent.label_ == 'CARDINAL':
            name = number_service.parse(ent.text)
            description = "number"
        elif ent.label_ == 'DATE':
            name = str(date_service.extractDate(ent.text))
            description = "date"
        elif ent.label_ == 'QUANTITY':
            name = number_service.parse(ent.text)
            description = "quantity"
        elif ent.label_ == 'TIME':
            name = ent.text
            description = "time"


        entity.append({
            "name": name,
            "type": ent.label_,
            "description": description
        })

    return entity

