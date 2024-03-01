
import dill as pickle

def startPickling():
    with open('PClass.pkl', 'wb') as file:
        pickle.dump(PClass,file)

    with open('getRes.pkl', 'wb') as file:
        pickle.dump(getRes,file)

    with open('newWords.pkl', 'wb') as file:
        pickle.dump(newWords,file)


    with open('ourClasses.pkl', 'wb') as file:
        pickle.dump(ourClasses,file)

    with open('data.pkl', 'wb') as file:
        pickle.dump(data,file)

    with open('wordBag.pkl', 'wb') as file:
        pickle.dump(wordBag,file)

    with open('ourText.pkl', 'wb') as file:
        pickle.dump(ourText,file)

    getRes_pkl=pickle.dumps(getRes)
    newWords_pkl=pickle.dumps(newWords) 
    ourClasses_pkl=pickle.dumps(ourClasses) 
    data_pkl=pickle.dumps(data) 

