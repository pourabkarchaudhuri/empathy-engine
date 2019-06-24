from textblob import TextBlob
import math

def sentiment_analyzer(text):
    inferenced_polarity = ""
    inferenced_subjectivity = ""

    testimonial = TextBlob(text)
    print("Polarity : {}".format(round(testimonial.sentiment.polarity, 2)))
    polarity = round(testimonial.sentiment.polarity, 2)
    print("Subjectivity : {}".format(round(testimonial.sentiment.subjectivity, 2)))
    subjectivity = round(testimonial.sentiment.subjectivity, 2)

    if polarity==0.0:
        inferenced_polarity = 'neutral'
        # print("Negative")
    elif polarity<0.65 and polarity>0.25:
        # print("Neutral")
        inferenced_polarity = 'supportive'
    elif polarity>=0.65:
        inferenced_polarity = 'positive'
        # print("Positive")
    else:
        inferenced_polarity = 'negative'
        # print("Unrecognized")

    if subjectivity<0.45:
        inferenced_subjectivity = 'objective'
        # print("Objective")
    elif subjectivity<0.65 and subjectivity>0.45:
        inferenced_subjectivity = 'neutral'
        # print("Neutral")
    elif subjectivity>=0.65:
        inferenced_subjectivity = 'subjective'
        # print("Subjective")
    else:
        inferenced_subjectivity = 'unrecognized'
        # print("Unrecognized")
    
    response = {
            'polarity': inferenced_polarity,
            'subjectivity': inferenced_subjectivity
        }
    return response
    


