import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from io import BytesIO


def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()



# Streamlit Page Config
st.set_page_config(
    page_title="Customer Churn Prediction App",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io',
        'Report a Bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'This app predicts customer churn using machine learning techniques. It analyzes customer data to identify at-risk customers and provides insights to enhance retention strategies.'
    }
)

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("customer_churn_model.keras")

model = load_model()

# Load the scaler
@st.cache_data
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

scaler = load_scaler()


def load_features():
    with open('selected_features.pkl','rb') as f:
        return pickle.load(f)
    
features = load_features() 



def to_excel_download(df, filename):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output
        




# -------------------- HOME PAGE --------------------
def show_home():
    
    st.title("📊 Customer Churn Prediction App")
    st.markdown("""
    Welcome to the **Customer Churn Prediction** application!

    🔍 This app uses a trained Artificial Neural Network (ANN) model to predict the likelihood of a customer churning.
    
    *Sections:*
    - Individual Prediction: Predict for a single customer.
    - Batch Prediction: Upload an Excel file to predict churn in bulk.
    - Visualization: See visual insights from uploaded data.
    - 👈🏽 Use the sidebar to navigate.
    """)





# ===== INDIVIDUAL CUSTOMER CHURN PREDICTION =====#

def individual_prediction():

    st.title('Individual Customer Churn Prediction')

    #input form
    with st.form(key="individual_form"):
        st.subheader("🔍 Enter Customer details to predict churn")

        senior_citizen= st.selectbox('Is the customer a senior citizen?', ["No", "Yes"])
        online_security = st.selectbox('Online Security?', ("No", "Yes"))
        partner = st.selectbox('Does the customer have a partner?', ["No", "Yes"])
        tech_support = st.selectbox('Do you have tech support?', ("No", "Yes"))
        paperless_billing = st.selectbox('Paperless Billing?', ["No", "Yes"])
        dependents = st.selectbox('Does the customer have dependents?', ["No", "Yes"])

        payment_method= st.radio('Payment Method:',[
            'Mailed check',
            'Electronic check',
            'Bank transfer (automatic)',
            'Credit card (automatic)',
        ])

        internet_service = st.radio("Internet Service type:", [
            'DSL', 
            'Fiber optic',
            'No'
        ])

        contract = st.radio("Contract type:", [
            'Month-to-month',
            'One year', 
            'Two year'
        ])

        tenure = st.number_input("Tenure (in months)", 0, 100, 12)
        monthly_charges = st.slider('Monthly Charges', min_value=18.25, max_value=118.0, step=0.25)

        submit = st.form_submit_button("🔍 Predict")


    if submit:
        #Label encode
        senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
        online_security_encoded = 1 if online_security == "Yes" else 0
        partner_encoded = 1 if partner == "Yes" else 0
        tech_support_encoded = 1 if tech_support == "Yes" else 0
        paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
        dependents_encoded = 1 if dependents == "Yes" else 0

    # One-hot encoding for payment method
        Payment_method_bank_transfer = 1 if payment_method == 'Bank transfer (automatic)' else 0
        Payment_method_credit_card = 1 if payment_method == 'Credit card (automatic)' else 0
        Payment_method_electronic_check= 1 if payment_method == 'Electronic check' else 0
        Payment_method_mailed_check= 1 if payment_method == 'Mailed check' else 0
    

    # One-hot encoding for internet service
        internet_service_DSL = 1 if internet_service == 'DSL' else 0
        internet_service_Fiber_optic = 1 if internet_service == 'Fiber optic' else 0
        Internet_Service_No= 1 if internet_service == 'No' else 0
    

    # One-hot encoding for contract
        contract_month= 1 if contract == 'Month-to-month' else 0
        contract_one= 1 if contract == 'One year' else 0
        contract_two= 1 if contract == 'Two year' else 0

    else:
        st.stop()    

    # Combine all features into one dictionary
    input_dict = {
        'Senior Citizen':senior_citizen_encoded,
        'Payment Method_Bank transfer (automatic)':Payment_method_bank_transfer,
        'Payment Method_Credit card (automatic)':Payment_method_credit_card,
        'Payment Method_Electronic check':Payment_method_electronic_check,
        'Payment Method_Mailed check':Payment_method_mailed_check,
        'Internet Service_DSL':internet_service_DSL,
        'Internet Service_Fiber optic':internet_service_Fiber_optic,
        'Internet Service_No':Internet_Service_No,
        'Contract_Month-to-month':contract_month,
        'Contract_One year':contract_one,
        'Contract_Two year':contract_two,
        'Online Security':online_security_encoded,
        'Partner': partner_encoded,
        'Tech Support': tech_support_encoded,
        'Monthly Charges': monthly_charges,
        'Paperless Billing': paperless_billing_encoded,
        'Dependents': dependents_encoded,
        'Tenure Months': tenure,
    
    }


    # Ensure all selected features are present
    for feature in features:
        if feature not in input_dict:
            input_dict[feature] = 0  # set missing one-hot features to 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0][0]
    st.success(f"Churn Probability: {prediction:.2%}")

    if prediction > 0.5:
        st.warning("This customer is likely to churn.")
    else:
        st.info("This customer is likely to stay.")





# ===== BATCH PREDICTION =====#

def batch_prediction():
    st.title('Batch Customer Churn Prediction')

    st.markdown("""
    Upload an Excel or CSV file containing customer data. The app will process the data and return churn predictions.
    
    *Instructions:*
    - File format: .xlsx or .csv
    - Make sure the columns match the model's expected input features.

    """)


    selected_features=['Senior Citizen', 'Tenure Months', 'Payment Method', 'Contract',
    'Internet Service', 'Partner', 'Tech Support', 'Monthly Charges',
    'Paperless Billing', 'Dependents', 'Online Security']

    # File uploader for CSV or Excel file
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    # Process uploaded file
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Display a preview of the raw data
        st.subheader("Raw Data Preview")
        st.dataframe(data.head())

     # ==== Feature Check ====
        missing_features = [col for col in selected_features if col not in data.columns]

        if missing_features:
           st.error("🚫 Missing required columns for prediction.")
           st.stop()  # Stop further execution if required columns are missing

        # ==== Proceed with prediction logic ====
        st.success("✅ All required columns are present. Proceeding with predictions...")    

        # ---- Generate Visualizations ----
        st.subheader("Visual Insights")

        # Initialize the list to store the figures
        fig_list = []

        # 1. Contract Distribution
        if 'Contract' in data.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='Contract', ax=ax)
            ax.set_title("Contract Type Distribution")
            st.pyplot(fig)
            fig_list.append(("contract_countplot.png", fig_to_bytes(fig)))

        # 2. Gender Distribution
        if 'Gender' in data.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='Gender', ax=ax)
            ax.set_title("Gender Distribution")
            st.pyplot(fig)
            fig_list.append(("gender_countplot.png", fig_to_bytes(fig)))

        # 3. Monthly Charges Histogram
        if 'Monthly Charges' in data.columns and 'Churn Value' in data.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(data=data, x="Monthly Charges", hue="Churn Value", fill=True, ax=ax)
            ax.set_title("Monthly Charges Distribution by Churn")
            st.pyplot(fig)
            fig_list.append(("monthly_charges_kde.png", fig_to_bytes(fig)))

        # 4. Tenure Histogram
        if 'Tenure Months' in data.columns:
            fig, ax = plt.subplots()
            sns.histplot(data['Tenure Months'].dropna(), kde=True, ax=ax)
            ax.set_title("Tenure Distribution")
            st.pyplot(fig)
            fig_list.append(("tenure_histogram.png", fig_to_bytes(fig)))

        if 'Churn Value' in data.columns and 'Contract' in data.columns:
            fig, ax = plt.subplots()
            contract_churn = data.groupby('Contract')['Churn Value'].value_counts(normalize=True).unstack().fillna(0)
            contract_churn.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
            ax.set_ylabel("Proportion")
            ax.set_title("Churn Rate by Contract Type")
            ax.legend(title="Churn", loc='upper right')
            st.pyplot(fig)
            fig_list.append(("churn_by_contract.png", fig_to_bytes(fig)))

        if 'Senior Citizen' in data.columns and 'Churn Value' in data.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=data, x='Senior Citizen', hue='Churn Value', ax=ax)
            ax.set_title("Churn Count by Senior Citizen Status")
            st.pyplot(fig)
            fig_list.append(("churn_by_senior.png", fig_to_bytes(fig)))


        if 'Payment Method' in data.columns and 'Churn Value' in data.columns:
           fig, ax = plt.subplots()
           churn_by_payment = data.groupby('Payment Method')['Churn Value'].value_counts(normalize=True).unstack().fillna(0)
           churn_by_payment.plot(kind='bar', stacked=True, ax=ax, colormap='Accent')
           ax.set_ylabel("Proportion")
           ax.set_title("Churn by Payment Method")
           ax.legend(title="Churn")
           st.pyplot(fig)
           fig_list.append(("churn_by_payment_method.png", fig_to_bytes(fig)))    



        # Download All Visualizations
        st.subheader("⬇️ Download All Visualizations")
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for name, fig_bytes in fig_list:
                zip_file.writestr(name, fig_bytes)
                zip_buffer.seek(0)
        st.download_button("Download Visualizations as ZIP", data=zip_buffer, file_name="visualizations.zip")


    else:
        st.error("Upload a file for prediction👆🏽.")
        st.stop()
    



   #--Making prediction reports---#
    
    data_for_prediction=data[selected_features]

    # Binary Label Encoding
    binary_cols = ['Senior Citizen','Partner', 'Tech Support', 'Paperless Billing', 'Dependents', 'Online Security']
    binary_map={'Yes': 1, 'No': 0}

    for col in binary_cols:
        data_for_prediction[col]=data_for_prediction[col].map(binary_map)
        data_for_prediction[col]=data_for_prediction[col].fillna(0).astype(int)

    # One-hot encode multi-class categorical columns
    multi_cat_cols = ['Payment Method', 'Contract', 'Internet Service']
    data_for_prediction= pd.get_dummies(data_for_prediction, columns=multi_cat_cols)

    col_to_convert=['Payment Method_Bank transfer (automatic)','Payment Method_Credit card (automatic)',
                    'Payment Method_Electronic check','Payment Method_Mailed check',
                    'Internet Service_DSL','Internet Service_Fiber optic','Internet Service_No',
                    'Contract_Month-to-month','Contract_One year','Contract_Two year']
    data_for_prediction[col_to_convert]=data_for_prediction[col_to_convert].astype(int)

    
    for col in features:
        if col not in data_for_prediction.columns:
            data_for_prediction[col]=0

    data_for_prediction=data_for_prediction[features]   
    
    
       # Scale numerical features
    X_scaled = scaler.transform(data_for_prediction)

    # Predict churn
    predictions = model.predict(X_scaled)
    predicted_labels= (predictions > 0.5).astype(int).flatten()

    # Add results back
    data_for_prediction['Churn Predictions']=np.where(predicted_labels==1,'Churn','Retain')
    

    #Separate results
    churned=data[data_for_prediction['Churn Predictions']=='Churn']
    retained=data[data_for_prediction['Churn Predictions']=='Retain']
    
    
    st.subheader("Customers Likely to Churn")
    churned_with_details = pd.merge(churned, data, on='CustomerID')
    st.dataframe(churned_with_details)
    
    st.subheader("Customers Likely to Stay")
    retained_with_details = pd.merge(retained, data, on='CustomerID')
    st.dataframe(retained_with_details)


    # Create a zip for download

    def df_to_excel_bytes(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
            output.seek(0)
        return output.getvalue()

        # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("churned_customers.xlsx", df_to_excel_bytes(churned))
        zip_file.writestr("retained_customers.xlsx", df_to_excel_bytes(retained))
        zip_buffer.seek(0)

    st.download_button(
        label="Download Prediction Reports (ZIP)",
        data=zip_buffer,
        file_name="customer_churn_reports.zip",
        mime="application/zip")
    
    
# Sidebar menu
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["🏠 Home", "👤 Individual Prediction", "📂 Batch Prediction"])

if selection=="🏠 Home":
    show_home()

elif selection == "👤 Individual Prediction":
    individual_prediction()

elif selection=="📂 Batch Prediction":
    batch_prediction()     
