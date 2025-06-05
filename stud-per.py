import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler , LabelEncoder

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
#To execute the above url function line 
if __name__ == "__main__" :
    main()