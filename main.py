import streamlit as st
import pandas as pd
import pickle
import os


# Load the pre-trained model
current_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(current_dir, "food_order_pred_log.pkl")
with open(pickle_file_path, "rb") as f:
    model = pickle.load(f)

# Define the function to preprocess input data and make predictions
def predict_order(Age, Gender, Marital_Status, Occupation, Monthly_Income, Educational_Qualifications, Family_size, Pin_code, Response):
    # Convert categorical inputs to numerical
    gender_mapping = {"Female":0, "Male":1}
    marital_mapping = {"Single":1, "Married":2, "Not Revealed":3}
    occupation_mapping = {"Student":1, "Employee":2, "Self Employed":3, "Housewife":4}
    edu_mapping = {"Graduate":1, "Post Graduate":2, "Ph.D":3, "School":4, "Uneducated":5}
    response_mapping = {"Negative":0, "Positive":1}

    Gender = gender_mapping[Gender]
    Marital_Status = marital_mapping[Marital_Status]
    Occupation = occupation_mapping[Occupation]
    Educational_Qualifications = edu_mapping[Educational_Qualifications]
    Response = response_mapping[Response]

    # Create features array
    features = pd.DataFrame([[Age, Gender, Marital_Status, Occupation, Monthly_Income, Educational_Qualifications, Family_size, Pin_code,Response]],columns=['Age', 'Gender', 'Marital_Status', 'Occupation', 'Monthly_Income', 'Educational_Qualifications', 'Family_size', 'Pin_code','Response'])

    # Make prediction
    prediction = model.predict(features)[0]
    return prediction

# Streamlit app
def main():
    st.title("🛒Online Food Order Prediction")
    st.write("Enter Customer Details to Predict If the Customer Will Order Again")

    # User input for customer details
    Age = st.number_input("Enter the Age of the Customer:")
    Gender = st.selectbox("Gender of the Customer", ['Female', 'Male'])
    Marital_Status = st.selectbox("Marital Status of the Customer", ['Single', 'Married', 'Not Revealed'])
    Occupation = st.selectbox("Occupation of the Customer", ["Student", "Employee", "Self Employed", "Housewife"])
    Monthly_Income = st.number_input("Monthly Income:")
    Educational_Qualifications = st.selectbox("Educational Qualification", ["Graduate", "Post Graduate", "Ph.D", "School", "Uneducated"])
    Family_size = st.number_input("Family Size:")
    Pin_code = st.number_input("Pin Code:")
    Response = st.selectbox("Response of the Last Order", ["Negative", "Positive"])

    # Make prediction when 'Predict' button is clicked
    if st.button("Predict"):
        prediction_result = predict_order(Age, Gender, Marital_Status, Occupation, Monthly_Income, Educational_Qualifications, Family_size, Pin_code,Response)
        if prediction_result==0:
            st.write("The Customer won't order again")
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQvpZjQxCuSYHpt69wdhdl5iZ7b_4I4bC9GVg&s')
        if prediction_result==1:
            st.write("The Customer will order again")
            st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYKzdLkyMj8QVma3gWIg9qGqrIzaeWj9qcKw&s')

if __name__ == "__main__":
    main()


