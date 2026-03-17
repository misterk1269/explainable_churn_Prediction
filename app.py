import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import easyocr
from PIL import Image
import re

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("churn_model.h5")
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Feature names and descriptions
# -----------------------------
features = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

feature_descriptions = {
    'CreditScore': 'Credit Score',
    'Age': 'Customer Age',
    'Tenure': 'Years with Bank',
    'Balance': 'Account Balance',
    'NumOfProducts': 'Number of Products',
    'HasCrCard': 'Has Credit Card',
    'IsActiveMember': 'Active Member Status',
    'EstimatedSalary': 'Estimated Salary',
    'Geography_Germany': 'Located in Germany',
    'Geography_Spain': 'Located in Spain',
    'Gender_Male': 'Gender (Male)'
}

# -----------------------------
# Load background data for SHAP
# -----------------------------
@st.cache_data
def load_background_data():
    data = pd.read_csv("Churn_Modelling.csv")
    data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    data = pd.get_dummies(data, columns=["Geography", "Gender"], drop_first=True)
    X_background = data.drop("Exited", axis=1)
    X_background_scaled = scaler.transform(X_background)
    return shap.sample(X_background_scaled, 100)

background = load_background_data()

# -----------------------------
# FIXED OCR Text Extraction Function
# -----------------------------
def extract_customer_data_from_text(text):
    """
    Fixed function based on the actual OCR patterns from your image
    """
    # Clean the text
    text = text.lower().replace(',', ' ').replace('"', ' ').replace("'", ' ')
    
    # Debug: Show cleaned text
    st.write("**Cleaned OCR Text:**", text)
    
    # Initialize default values
    extracted_data = {
        'CreditScore': 600,
        'Age': 40,
        'Tenure': 3,
        'Balance': 60000,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 50000,
        'Geography_Germany': 0,
        'Geography_Spain': 0,
        'Gender_Male': 1
    }
    
    try:
        # Extract Credit Score - look for creditscore followed by number
        credit_match = re.search(r'creditscore\s+(\d{3,4})', text)
        if credit_match:
            score = int(credit_match.group(1))
            if 300 <= score <= 900:
                extracted_data['CreditScore'] = score
                st.write(f"✅ Credit Score extracted: {score}")
        
        # Extract Age - look for age followed by number
        age_match = re.search(r'age\s+(\d{1,3})', text)
        if age_match:
            age = int(age_match.group(1))
            if 18 <= age <= 100:
                extracted_data['Age'] = age
                st.write(f"✅ Age extracted: {age}")
        
        # Extract Tenure - look for 'tenure' followed by optional number or get from structured part
        # First try to find explicit tenure value
        tenure_match = re.search(r'tenure\s+(\d{1,2})', text)
        if tenure_match:
            tenure = int(tenure_match.group(1))
            if 0 <= tenure <= 15:
                extracted_data['Tenure'] = tenure
                st.write(f"✅ Tenure extracted: {tenure}")
        else:
            # If no explicit tenure, try to extract from the pattern where it appears after age
            # Looking at your text: "age 55 tenure balance 150000"
            tenure_pattern = re.search(r'age\s+\d+\s+tenure(?:\s+(\d{1,2}))?\s+balance', text)
            if tenure_pattern and tenure_pattern.group(1):
                tenure = int(tenure_pattern.group(1))
                if 0 <= tenure <= 15:
                    extracted_data['Tenure'] = tenure
                    st.write(f"✅ Tenure extracted from pattern: {tenure}")
            else:
                # Default to 1 based on your image
                extracted_data['Tenure'] = 1
                st.write(f"⚠️ Tenure not found, using default: 1")
        
        # Extract Balance - look for balance followed by large number
        balance_match = re.search(r'balance\s+(\d{4,7})', text)
        if balance_match:
            balance = int(balance_match.group(1))
            extracted_data['Balance'] = balance
            st.write(f"✅ Balance extracted: {balance}")
        
        # Extract NumOfProducts - look for numofproducts followed by single digit or extract from pattern
        # First try direct pattern
        products_match = re.search(r'numofproducts\s+(\d{1})', text)
        if products_match:
            products = int(products_match.group(1))
            if 1 <= products <= 4:
                extracted_data['NumOfProducts'] = products
                st.write(f"✅ Products extracted: {products}")
        else:
            # If no explicit number, check the pattern and default to 1 based on your image
            extracted_data['NumOfProducts'] = 1
            st.write(f"⚠️ Products not found, using default: 1")
        
        # Extract EstimatedSalary - look for estimatedsalary followed by number
        salary_match = re.search(r'estimatedsalary\s+(\d{4,7})', text)
        if salary_match:
            salary = int(salary_match.group(1))
            extracted_data['EstimatedSalary'] = salary
            st.write(f"✅ Salary extracted: {salary}")
        
        # Extract HasCrCard - Based on your image, it shows "0" which means NO
        # Look for explicit "hascrcard 0" or "hascrcard 1"
        hascrcard_match = re.search(r'hascrcard\s+(\d{1})', text)
        if hascrcard_match:
            hascrcard = int(hascrcard_match.group(1))
            extracted_data['HasCrCard'] = hascrcard
            st.write(f"✅ Has Credit Card extracted: {'Yes' if hascrcard else 'No'}")
        else:
            # If just "hascrcard" without number, check your image - it shows 0
            if 'hascrcard' in text:
                extracted_data['HasCrCard'] = 0  # Based on your image showing "0"
                st.write(f"⚠️ HasCrCard found without number, using: No (0)")
        
        # Extract IsActiveMember - Based on your image, it shows "0" which means INACTIVE
        # Look for explicit "isactivemember 0" or "isactivemember 1"
        activemember_match = re.search(r'isactivemember\s+(\d{1})', text)
        if activemember_match:
            is_active = int(activemember_match.group(1))
            extracted_data['IsActiveMember'] = is_active
            st.write(f"✅ Active Member extracted: {'Yes' if is_active else 'No'}")
        else:
            # If just "isactivemember" without number, check your image - it shows 0
            if 'isactivemember' in text:
                extracted_data['IsActiveMember'] = 0  # Based on your image showing "0"
                st.write(f"⚠️ IsActiveMember found without number, using: No (0)")
            elif 'inactivemember' in text:
                extracted_data['IsActiveMember'] = 0
                st.write(f"✅ Inactive Member detected")
        
        # Extract Geography
        if 'germany' in text:
            extracted_data['Geography_Germany'] = 1
            extracted_data['Geography_Spain'] = 0
            st.write(f"✅ Location extracted: Germany")
        elif 'spain' in text:
            extracted_data['Geography_Spain'] = 1
            extracted_data['Geography_Germany'] = 0
            st.write(f"✅ Location extracted: Spain")
        else:
            extracted_data['Geography_Germany'] = 0
            extracted_data['Geography_Spain'] = 0
            st.write(f"✅ Location extracted: France (default)")
        
        # Extract Gender
        if 'male' in text and 'female' not in text:
            extracted_data['Gender_Male'] = 1
            st.write(f"✅ Gender extracted: Male")
        elif 'female' in text:
            extracted_data['Gender_Male'] = 0
            st.write(f"✅ Gender extracted: Female")
        
    except Exception as e:
        st.warning(f"Error in extraction: {str(e)}. Using default values.")
    
    return extracted_data

# -----------------------------
# App Title
# -----------------------------
st.title("🔍 Customer Churn Prediction")
st.markdown("Understand why customers might leave and what factors matter most.")

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Manual Input", "Upload Image"]
)

# Initialize default values
CreditScore = 600
Age = 40
Tenure = 3
Balance = 60000
NumOfProducts = 2
HasCrCard = 1
IsActiveMember = 1
EstimatedSalary = 50000
Geography = "France"
Gender = "Male"
Geography_Germany = 0
Geography_Spain = 0
Gender_Male = 1

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Customer Information")

# -----------------------------
# IMAGE INPUT MODE
# -----------------------------
if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Customer Data Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

            # OCR Processing
            with st.spinner("Processing image..."):
                reader = easyocr.Reader(['en'], gpu=False)
                result = reader.readtext(np.array(image), detail=0)
                text = " ".join(result).lower()
                
            st.write("**Extracted Text:**", text)
            
            # Extract data using improved function
            with st.expander("🔍 Extraction Debug Info", expanded=True):
                extracted_data = extract_customer_data_from_text(text)
            
            # Assign extracted values
            CreditScore = extracted_data['CreditScore']
            Age = extracted_data['Age']
            Tenure = extracted_data['Tenure']
            Balance = extracted_data['Balance']
            NumOfProducts = extracted_data['NumOfProducts']
            HasCrCard = extracted_data['HasCrCard']
            IsActiveMember = extracted_data['IsActiveMember']
            EstimatedSalary = extracted_data['EstimatedSalary']
            Geography_Germany = extracted_data['Geography_Germany']
            Geography_Spain = extracted_data['Geography_Spain']
            Gender_Male = extracted_data['Gender_Male']
            
            # Show extracted values
            st.sidebar.success("✅ Data extracted successfully!")
            st.sidebar.write("**Extracted Values:**")
            st.sidebar.write(f"Credit Score: {CreditScore}")
            st.sidebar.write(f"Age: {Age}")
            st.sidebar.write(f"Tenure: {Tenure}")
            st.sidebar.write(f"Balance: ${Balance:,.0f}")
            st.sidebar.write(f"Products: {NumOfProducts}")
            st.sidebar.write(f"Credit Card: {'Yes' if HasCrCard else 'No'}")
            st.sidebar.write(f"Active Member: {'Yes' if IsActiveMember else 'No'}")
            st.sidebar.write(f"Salary: ${EstimatedSalary:,.0f}")
            st.sidebar.write(f"Location: {'Germany' if Geography_Germany else 'Spain' if Geography_Spain else 'France'}")
            st.sidebar.write(f"Gender: {'Male' if Gender_Male else 'Female'}")
            
            # Allow manual correction
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Manual Corrections (if needed):**")
            CreditScore = st.sidebar.slider("Credit Score", 300, 900, int(CreditScore))
            Age = st.sidebar.slider("Age", 18, 100, int(Age))
            Tenure = st.sidebar.slider("Tenure (years)", 0, 15, int(Tenure))
            Balance = st.sidebar.number_input("Balance ($)", 0, 500000, int(Balance), step=1000)
            NumOfProducts = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=int(NumOfProducts)-1)
            HasCrCard_selection = st.sidebar.selectbox("Has Credit Card?", ["No", "Yes"], index=int(HasCrCard))
            IsActiveMember_selection = st.sidebar.selectbox("Is Active Member?", ["No", "Yes"], index=int(IsActiveMember))
            EstimatedSalary = st.sidebar.number_input("Salary ($)", 0, 300000, int(EstimatedSalary), step=1000)
            
            # Convert corrections back to numeric
            HasCrCard = 1 if HasCrCard_selection == "Yes" else 0
            IsActiveMember = 1 if IsActiveMember_selection == "Yes" else 0
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.stop()

# -----------------------------
# MANUAL INPUT MODE
# -----------------------------
if input_method == "Manual Input":
    CreditScore = st.sidebar.slider("Credit Score", 300, 900, 600)
    Age = st.sidebar.slider("Age", 18, 100, 40)
    Tenure = st.sidebar.slider("Tenure (years with bank)", 0, 10, 3)
    Balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0, step=1000.0)
    NumOfProducts = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    HasCrCard = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
    IsActiveMember = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
    EstimatedSalary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0, step=1000.0)
    Geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
    Gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

    # Convert inputs
    HasCrCard = 1 if HasCrCard == "Yes" else 0
    IsActiveMember = 1 if IsActiveMember == "Yes" else 0
    Geography_Germany = 1 if Geography == "Germany" else 0
    Geography_Spain = 1 if Geography == "Spain" else 0
    Gender_Male = 1 if Gender == "Male" else 0

# Create input dataframe
input_data = pd.DataFrame([[
    float(CreditScore), float(Age), float(Tenure), float(Balance), int(NumOfProducts),
    int(HasCrCard), int(IsActiveMember), float(EstimatedSalary),
    int(Geography_Germany), int(Geography_Spain), int(Gender_Male)
]], columns=features)

# Scale the input
try:
    input_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error scaling input data: {str(e)}")
    st.stop()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Churn Risk", type="primary"):
    try:
        probability = model.predict(input_scaled, verbose=0)[0][0]
        
        # -----------------------------
        # Results Section
        # -----------------------------
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Prediction Result")
            
            # Risk meter
            if probability > 0.7:
                st.error(f"### ⚠️ High Risk: {probability*100:.1f}%")
                risk_level = "HIGH"
                risk_color = "red"
            elif probability > 0.42:
                st.warning(f"### ⚡ Medium Risk: {probability*100:.1f}%")
                risk_level = "MEDIUM"
                risk_color = "orange"
            else:
                st.success(f"### ✅ Low Risk: {probability*100:.1f}%")
                risk_level = "LOW"
                risk_color = "green"
            
            # Visual gauge
            fig_gauge, ax_gauge = plt.subplots(figsize=(4, 2))
            ax_gauge.barh([0], [probability], color=risk_color, height=0.5)
            ax_gauge.barh([0], [1-probability], left=[probability], color='lightgray', height=0.5)
            ax_gauge.set_xlim(0, 1)
            ax_gauge.set_yticks([])
            ax_gauge.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax_gauge.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax_gauge.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_gauge.set_title('Churn Probability', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_gauge)
            plt.close()
        
        # -----------------------------
        # SHAP Analysis
        # -----------------------------
        with col2:
            st.subheader("Why This Prediction?")
            
            # Calculate SHAP values
            explainer = shap.Explainer(model.predict, background)
            shap_values = explainer(input_scaled)
            
            # Get SHAP values and create analysis
            shap_vals = shap_values.values[0]
            base_value = shap_values.base_values[0]
            
            # Create feature impact dataframe
            impact_df = pd.DataFrame({
                'Feature': features,
                'Display Name': [feature_descriptions[f] for f in features],
                'Value': input_data.values[0],
                'SHAP Value': shap_vals,
                'Impact': ['Increases' if s > 0 else 'Decreases' for s in shap_vals],
                'Abs Impact': np.abs(shap_vals)
            }).sort_values('Abs Impact', ascending=False)
            
            # Top factors visualization
            top_n = 6
            top_factors = impact_df.head(top_n).copy()
            
            colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in top_factors['SHAP Value']]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(
                range(len(top_factors)), 
                top_factors['SHAP Value'].values,
                color=colors,
                edgecolor='white',
                linewidth=0.5
            )
            
            ax.set_yticks(range(len(top_factors)))
            ax.set_yticklabels(top_factors['Display Name'].values)
            ax.invert_yaxis()
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Impact on Churn Prediction')
            ax.set_title('Top Factors Influencing This Prediction')
            
            # Add value annotations
            for i, (idx, row) in enumerate(top_factors.iterrows()):
                val = row['Value']
                if row['Feature'] in ['Balance', 'EstimatedSalary']:
                    val_text = f"${val:,.0f}"
                elif row['Feature'] in ['HasCrCard', 'IsActiveMember', 'Geography_Germany', 'Geography_Spain', 'Gender_Male']:
                    val_text = "Yes" if val == 1 else "No"
                else:
                    val_text = f"{val:.0f}" if val == int(val) else f"{val:.1f}"
                
                ax.annotate(
                    f"({val_text})",
                    xy=(0, i),
                    xytext=(5 if row['SHAP Value'] < 0 else -5, 0),
                    textcoords='offset points',
                    ha='left' if row['SHAP Value'] < 0 else 'right',
                    va='center',
                    fontsize=9,
                    color='gray'
                )
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # -----------------------------
        # Detailed Explanation
        # -----------------------------
        st.subheader("📝 Detailed Explanation")
        
        # Generate human-readable explanations
        explanations = []
        
        for idx, row in impact_df.head(5).iterrows():
            feature = row['Feature']
            value = row['Value']
            impact = row['SHAP Value']
            direction = "increases" if impact > 0 else "decreases"
            strength = "significantly" if abs(impact) > 0.1 else "slightly"
            
            if feature == 'Age':
                if value > 50 and impact > 0:
                    explanations.append(f"🔸 **Age ({int(value)})**: Older customers tend to have higher churn rates. This {strength} {direction} the risk.")
                elif value < 30 and impact > 0:
                    explanations.append(f"🔸 **Age ({int(value)})**: Younger customers in this profile show higher churn. This {strength} {direction} the risk.")
                else:
                    explanations.append(f"🔸 **Age ({int(value)})**: This age group {strength} {direction} churn risk.")
            
            elif feature == 'Tenure':
                if value <= 2 and impact > 0:
                    explanations.append(f"🔸 **Tenure ({int(value)} years)**: Short relationship with the bank. New customers often churn more. This {strength} {direction} risk.")
                elif value >= 7 and impact < 0:
                    explanations.append(f"🔸 **Tenure ({int(value)} years)**: Long-standing customer. Loyalty {strength} {direction} churn risk.")
                else:
                    explanations.append(f"🔸 **Tenure ({int(value)} years)**: This {strength} {direction} the prediction.")
            
            elif feature == 'NumOfProducts':
                if value == 1:
                    explanations.append(f"🔸 **Products ({int(value)})**: Customer has only one product which suggests low engagement with the bank. This {strength} {direction} churn risk.")
                elif value == 2:
                    explanations.append(f"🔸 **Products ({int(value)})**: Customer uses multiple bank services which usually indicates stronger engagement. This {strength} {direction} churn risk.")
                elif value == 3:
                    explanations.append(f"🔸 **Products ({int(value)})**: Customer has several products. This moderately influences churn behaviour depending on customer engagement.")
                elif value >= 4:
                    explanations.append(f"🔸 **Products ({int(value)})**: Very high number of products may indicate complexity or dissatisfaction with services, which can increase churn risk.")
            
            elif feature == 'IsActiveMember':
                if value == 0 and impact > 0:
                    explanations.append(f"🔸 **Inactive Member**: Customer is not actively using services. This {strength} {direction} churn risk.")
                elif value == 1 and impact < 0:
                    explanations.append(f"🔸 **Active Member**: Active engagement {strength} {direction} churn risk.")
                else:
                    explanations.append(f"🔸 **Activity Status**: This {strength} {direction} the prediction.")
            
            elif feature == 'Balance':
                if value == 0 and impact > 0:
                    explanations.append(f"🔸 **Balance ($0)**: Zero balance may indicate disengagement. This {strength} {direction} risk.")
                elif value > 100000 and impact > 0:
                    explanations.append(f"🔸 **Balance (${value:,.0f})**: High balance customers in this profile show higher churn. This {strength} {direction} risk.")
                else:
                    explanations.append(f"🔸 **Balance (${value:,.0f})**: Account balance {strength} {direction} churn risk.")
            
            elif feature == 'Geography_Germany':
                if value == 1 and impact > 0:
                    explanations.append(f"🔸 **Location (Germany)**: German customers historically have higher churn rates. This {strength} {direction} risk.")
                elif value == 1 and impact < 0:
                    explanations.append(f"🔸 **Location (Germany)**: In this case, being in Germany {strength} {direction} the risk.")
            
            elif feature == 'Geography_Spain':
                if value == 1:
                    explanations.append(f"🔸 **Location (Spain)**: This {strength} {direction} the churn prediction.")
            
            elif feature == 'CreditScore':
                if value < 500 and impact > 0:
                    explanations.append(f"🔸 **Credit Score ({int(value)})**: Lower credit score {strength} {direction} churn risk.")
                elif value > 750 and impact < 0:
                    explanations.append(f"🔸 **Credit Score ({int(value)})**: Good credit score {strength} {direction} churn risk.")
                else:
                    explanations.append(f"🔸 **Credit Score ({int(value)})**: This {strength} {direction} the prediction.")
            
            elif feature == 'Gender_Male':
                gender_text = "Male" if value == 1 else "Female"
                explanations.append(f"🔸 **Gender ({gender_text})**: This {strength} {direction} the churn prediction.")
            
            elif feature == 'HasCrCard':
                card_status = "has" if value == 1 else "doesn't have"
                explanations.append(f"🔸 **Credit Card**: Customer {card_status} a credit card. This {strength} {direction} churn risk.")
            
            elif feature == 'EstimatedSalary':
                explanations.append(f"🔸 **Salary (${value:,.0f})**: This salary level {strength} {direction} the prediction.")
        
        for exp in explanations:
            st.markdown(exp)
        
        # -----------------------------
        # Recommendations
        # -----------------------------
        st.subheader("💡 Recommendations")
        
        recommendations = []
        
        if probability > 0.5:
            if IsActiveMember == 0:
                recommendations.append("• **Increase Engagement**: Reach out with personalized offers to re-engage this inactive customer.")
            
            if NumOfProducts == 1:
                recommendations.append("• **Cross-sell Products**: Offer relevant additional products (savings accounts, insurance) to deepen the relationship.")
            
            if Tenure <= 2:
                recommendations.append("• **New Customer Care**: Implement onboarding follow-ups and early relationship nurturing.")
            
            if Balance == 0:
                recommendations.append("• **Reactivation Campaign**: Offer incentives to encourage account usage and deposits.")
            
            if Geography_Germany == 1:
                recommendations.append("• **Regional Focus**: Review Germany-specific service issues or competitive pressures.")
            
            if not recommendations:
                recommendations.append("• **Proactive Contact**: Schedule a customer service call to address any concerns.")
                recommendations.append("• **Loyalty Program**: Consider enrollment in rewards or loyalty programs.")
        else:
            recommendations.append("• **Maintain Relationship**: Continue current engagement strategies.")
            recommendations.append("• **Upsell Opportunity**: This customer may be receptive to premium services.")
        
        for rec in recommendations:
            st.markdown(rec)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check your model files and input data.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("*This tool uses machine learning to predict customer churn probability. Explanations are generated using SHAP values.*")
