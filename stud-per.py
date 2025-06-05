import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler , LabelEncoder

#For coonecting to mongodb to store the user data and output result
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://Anurag:Anurag1234@cluster0.bagh3cm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["Student"]                                                              #creating the databse 
collections = db["Student_prediction"]                                              #creating the collection variable to store the data



#writing the function to load the model 
def load_model():
    with open("Student_LR_Final_Model.pkl","rb") as file:
        model,scaller, le = pickle.load(file)
        return model , scaller , le

#writing a function where user fill thier choice of data and then get converted into our tranformed data i.e user fill yes or no and we get 1 or 0 
def preprocessing_input_data(data , scaller , le) :
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transform = scaller.transform(df)
    return df_transform

#writing a predict function 
def predict_data(data):
    model,scaller, le = load_model()
    process_data = preprocessing_input_data(data,scaller, le)
    prediction = model.predict(process_data)
    return prediction

#Creating the url for app 
def main():
    st.title("Student Performance prediction")
    st.write("enter your data to get a prediction for your performance")

    #now we are going to create a field where user can fill the data
    hour_study = st.number_input("Hours Studied", min_value = 1, max_value = 10 , value = 5)
    previous_score = st.number_input("Previous Scores", min_value = 1, max_value = 100 , value = 70)
    extr = st.selectbox("Extracurricular Activities",["Yes" , "No"])                                                                              #Point to be noted yes awr no kaa starting letter same hona chahiye nhi to error aa jayega 
    sleeping_hour = st.number_input("Sleep Hours", min_value = 4, max_value = 10 , value = 7)
    num_of_paper_solved = st.number_input("Sample Question Papers Practiced", min_value = 0, max_value = 100 , value = 5)

    #Now we are going to create a button at the bottom which when click send all the user input into the model
    if st.button("predict-Your-score") :
        user_data = {
            "Hours Studied":hour_study,
            "Previous Scores":previous_score, 
            "Extracurricular Activities":extr,
            "Sleep Hours":sleeping_hour,
            "Sample Question Papers Practiced":num_of_paper_solved 
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")
        user_data["prediction"] = float(prediction)                                                                                                                            #user data me prediction data ko add kro 

        # Imagine you have a big box of toys (that's called user_data), and each toy has a name tag on it. But sometimes the name tags are written in a special computer language that's hard to read.
        # This code is like having a helpful friend who goes through your toy box and rewrites all the name tags in regular English so everyone can understand them better.
        user_data = {key: int(value) if isinstance(value , np.integer) else float(value) if isinstance(value , np.floating) else value for key , value in user_data.items()}   


        collections.insert_one(user_data)                                                                                                                                      #mongodb ke collections variable me store krdo pure data ko 
#To execute the above url function line 
if __name__ == "__main__" :
    main()