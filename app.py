import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

class StreamlitIncomePredictor:
    def __init__(self):
        st.set_page_config(
            page_title="Income Prediction App",
            page_icon="💰",
            layout="wide"
        )
        
        st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    def load_model(self):
        """Load the trained model"""
        try:
            with open('ensemble_model.pkl', 'rb') as file:
                self.model = pickle.load(file)
            return True
        except FileNotFoundError:
            st.error("Model file not found. Please ensure the model is trained and saved.")
            return False

    def create_input_form(self):
        """Create the input form matching the exact preprocessed features"""
        st.title("Income Prediction App 💰")
        
        st.markdown("""
        ## Adult Income Prediction
        This application predicts income levels based on various personal and professional attributes.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Basic Information")
            age = st.number_input("Age", min_value=17, max_value=90, value=30)
            fnlwgt = st.number_input("Final Weight", min_value=0, value=200000)
            education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
            capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
            capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

        with col2:
            st.subheader("Work Information")
            workclass = st.selectbox("Workclass", 
                ['Private', 'Federal-gov', 'Local-gov', 'Never-worked', 
                 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'])
            
            education = st.selectbox("Education",
                ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
                 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
                 'Masters', 'Preschool', 'Prof-school', 'Some-college'])
            
            occupation = st.selectbox("Occupation",
                ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
                 'Other-service', 'Priv-house-serv', 'Prof-specialty',
                 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'])

        with col3:
            st.subheader("Personal Information")
            marital_status = st.selectbox("Marital Status",
                ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])
            
            relationship = st.selectbox("Relationship",
                ['Husband', 'Not-in-family', 'Other-relative', 'Own-child',
                 'Unmarried', 'Wife'])
            
            race = st.selectbox("Race",
                ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'])
            
            sex = st.selectbox("Sex", ['Female', 'Male'])

            native_country = st.selectbox("Native Country",
                ['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba',
                 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England',
                 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti',
                 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary',
                 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan',
                 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
                 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico',
                 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago',
                 'United-States', 'Vietnam', 'Yugoslavia'])

        return {
            'age': age,
            'fnlwgt': fnlwgt,
            'education-num': education_num,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'workclass': workclass,
            'education': education,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'native-country': native_country
        }

    def add_derived_features(self, data):
        """Add age groups and hours groups features"""
        # Age groups
        age = data['age'].iloc[0]
        data['age_group_VeryYoung'] = 1 if age < 25 else 0
        data['age_group_Young'] = 1 if 25 <= age < 35 else 0
        data['age_group_Middle'] = 1 if 35 <= age < 50 else 0
        data['age_group_Senior'] = 1 if 50 <= age < 65 else 0
        data['age_group_VerySenior'] = 1 if age >= 65 else 0

        # Hours groups
        hours = data['hours-per-week'].iloc[0]
        data['hours_group_PartTime'] = 1 if hours < 35 else 0
        data['hours_group_Regular'] = 1 if 35 <= hours <= 40 else 0
        data['hours_group_Overtime'] = 1 if 40 < hours <= 60 else 0
        data['hours_group_Heavy'] = 1 if hours > 60 else 0
        
        return data

    def preprocess_input(self, input_data):
        """Preprocess input data to match the exact training data format"""
        # Initialize a dictionary with all features set to 0
        processed_data = {
            'age': input_data['age'],
            'fnlwgt': input_data['fnlwgt'],
            'education-num': input_data['education-num'],
            'capital-gain': input_data['capital-gain'],
            'capital-loss': input_data['capital-loss'],
            'hours-per-week': input_data['hours-per-week']
        }
    
    # Add binary columns for categorical variables with exact column names
    # Workclass
        workclass_prefix = 'workclass_ '
        for workclass in ['?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private',
                         'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']:
            processed_data[workclass_prefix + workclass] = 1 if input_data['workclass'] == workclass else 0

    # Education
        education_prefix = 'education_ '
        for edu in ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
                   'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
                   'Masters', 'Preschool', 'Prof-school', 'Some-college']:
            processed_data[education_prefix + edu] = 1 if input_data['education'] == edu else 0

    # Marital Status
        marital_prefix = 'marital-status_ '
        for status in ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                       'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']:
            processed_data[marital_prefix + status] = 1 if input_data['marital-status'] == status else 0

    # Occupation
        occupation_prefix = 'occupation_ '
        occupations = ['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                       'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
                       'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
                       'Tech-support', 'Transport-moving']
        for occ in occupations:
            processed_data[occupation_prefix + occ] = 1 if input_data['occupation'] == occ else 0

    # Relationship
        relationship_prefix = 'relationship_ '
        for rel in ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']:
            processed_data[relationship_prefix + rel] = 1 if input_data['relationship'] == rel else 0

    # Race
        race_prefix = 'race_ '
        for race in ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']:
            processed_data[race_prefix + race] = 1 if input_data['race'] == race else 0

    # Sex
        sex_prefix = 'sex_ '
        for sex in ['Female', 'Male']:
            processed_data[sex_prefix + sex] = 1 if input_data['sex'] == sex else 0

    # Native Country
        country_prefix = 'native-country_ '
        countries = ['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba',
                     'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England',
                     'France', 'Germany', 'Greece', 'Guatemala', 'Haiti',
                     'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary',
                     'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan',
                     'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)',
                     'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico',
                     'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago',
                     'United-States', 'Vietnam', 'Yugoslavia']
        for country in countries:
            processed_data[country_prefix + country] = 1 if input_data['native-country'] == country else 0

    # Convert to DataFrame
        df = pd.DataFrame([processed_data])
    
    # Add derived features
        df = self.add_derived_features(df)
    
    # Ensure all columns are present and in the correct order
        expected_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 
                       'hours-per-week', 'workclass_ ?', 'workclass_ Federal-gov', 
                       'workclass_ Local-gov', 'workclass_ Never-worked', 'workclass_ Private',
                       'workclass_ Self-emp-inc', 'workclass_ Self-emp-not-inc', 
                       'workclass_ State-gov', 'workclass_ Without-pay', 'education_ 10th',
                       'education_ 11th', 'education_ 12th', 'education_ 1st-4th',
                       'education_ 5th-6th', 'education_ 7th-8th', 'education_ 9th',
                       'education_ Assoc-acdm', 'education_ Assoc-voc', 'education_ Bachelors',
                       'education_ Doctorate', 'education_ HS-grad', 'education_ Masters',
                       'education_ Preschool', 'education_ Prof-school', 'education_ Some-college',
                       'marital-status_ Divorced', 'marital-status_ Married-AF-spouse',
                       'marital-status_ Married-civ-spouse', 'marital-status_ Married-spouse-absent',
                       'marital-status_ Never-married', 'marital-status_ Separated',
                       'marital-status_ Widowed', 'occupation_ ?', 'occupation_ Adm-clerical',
                       'occupation_ Armed-Forces', 'occupation_ Craft-repair',
                       'occupation_ Exec-managerial', 'occupation_ Farming-fishing',
                       'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct',
                       'occupation_ Other-service', 'occupation_ Priv-house-serv',
                       'occupation_ Prof-specialty', 'occupation_ Protective-serv',
                       'occupation_ Sales', 'occupation_ Tech-support',
                       'occupation_ Transport-moving', 'relationship_ Husband',
                       'relationship_ Not-in-family', 'relationship_ Other-relative',
                       'relationship_ Own-child', 'relationship_ Unmarried',
                       'relationship_ Wife', 'race_ Amer-Indian-Eskimo',
                       'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other',
                       'race_ White', 'sex_ Female', 'sex_ Male', 'native-country_ ?',
                       'native-country_ Cambodia', 'native-country_ Canada',
                       'native-country_ China', 'native-country_ Columbia',
                       'native-country_ Cuba', 'native-country_ Dominican-Republic',
                       'native-country_ Ecuador', 'native-country_ El-Salvador',
                       'native-country_ England', 'native-country_ France',
                       'native-country_ Germany', 'native-country_ Greece',
                       'native-country_ Guatemala', 'native-country_ Haiti',
                       'native-country_ Holand-Netherlands', 'native-country_ Honduras',
                       'native-country_ Hong', 'native-country_ Hungary',
                       'native-country_ India', 'native-country_ Iran',
                       'native-country_ Ireland', 'native-country_ Italy',
                       'native-country_ Jamaica', 'native-country_ Japan',
                       'native-country_ Laos', 'native-country_ Mexico',
                       'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)',
                       'native-country_ Peru', 'native-country_ Philippines',
                       'native-country_ Poland', 'native-country_ Portugal',
                       'native-country_ Puerto-Rico', 'native-country_ Scotland',
                       'native-country_ South', 'native-country_ Taiwan',
                       'native-country_ Thailand', 'native-country_ Trinadad&Tobago',
                       'native-country_ United-States', 'native-country_ Vietnam',
                       'native-country_ Yugoslavia',
                       'age_group_VeryYoung', 'age_group_Young', 'age_group_Middle',
                       'age_group_Senior', 'age_group_VerySenior',
                       'hours_group_PartTime', 'hours_group_Regular',
                       'hours_group_Overtime', 'hours_group_Heavy']
    
    # Add any missing columns with 0s
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
            
    # Ensure columns are in the correct order
        df = df[expected_columns]
    
        return df

    def predict(self, input_data):
        """Make prediction using the model"""
        processed_data = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        return prediction[0], probability[0]

    def run(self):
        """Run the Streamlit app"""
        if not self.load_model():
            return
        
        input_data = self.create_input_form()
        
        if st.button("Predict Income"):
            try:
                prediction, probability = self.predict(input_data)
                
                st.markdown("### Prediction Results")
                
                if prediction == 1:
                    st.success(f"Income Prediction: >50K (Probability: {probability[1]:.2%})")
                else:
                    st.info(f"Income Prediction: ≤50K (Probability: {probability[0]:.2%})")
                
                confidence = probability[1] if prediction == 1 else probability[0]
                st.markdown(f"""
                    <div class="prediction-box" style="background-color: rgba(76, 175, 80, {confidence});">
                        Prediction Confidence: {confidence:.2%}
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your input values and try again.")

if __name__ == "__main__":
    app = StreamlitIncomePredictor()
    app.run()