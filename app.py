import streamlit as st
import pandas as pd
import joblib
import json

# Set page configuration
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the app's appearance
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stSelectbox, .stNumber_input {
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    try:
        # Load the model
        model = joblib.load('model_pipeline.pkl')
        # Define top 15 frequent locations
        locations = [
            "Whitefield", "Electronic City", "Kanakpura Road", "Thanisandra",
            "Yelahanka", "Uttarahalli", "Hebbal", "Marathahalli", "Raja Rajeshwari Nagar",
            "Bannerghatta Road", "Hennur Road", "7th Phase JP Nagar", "Haralur Road", "Electronic City Phase II"
        ]
        return model, locations
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        return None, None

def predict_price(location, sqft, bath, bhk, model):
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'location': [location],
            'total_sqft': [float(sqft)],
            'bath': [float(bath)],
            'bhk': [float(bhk)]
        })
        # Make prediction
        prediction = model.predict(input_data)
        return float(prediction[0])
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # Load model and data
    model, locations = load_model_and_data()
    
    if model is None or locations is None:
        st.error("Failed to load necessary components. Please check the model and data files.")
        return

    # Header
    st.title("🏠 Bengaluru House Price Predictor")
    st.markdown("""
    This app predicts house prices in Bengaluru based on location, size, and amenities.
    Enter your requirements below to get an estimated price!
    """)

    # Create two columns for input
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Property Details")
        
        # Location selector with search
        location = st.selectbox(
            "Select Location",
            options=locations,
            help="Choose the area where you want to buy a house"
        )
        
        # Property features
        total_sqft = st.number_input(
            "Total Square Feet",
            min_value=100,
            max_value=2200,
            value=1000,
            help="Enter the total area in square feet"
        )
        
        # Create two columns for BHK and Bathrooms
        bhk_bath_col1, bhk_bath_col2 = st.columns(2)
        
        with bhk_bath_col1:
            bhk = st.number_input(
                "Number of Bedrooms (BHK)",
                min_value=1,
                max_value=10,
                value=2,
                help="Enter the number of bedrooms"
            )
            
        with bhk_bath_col2:
            bath = st.number_input(
                "Number of Bathrooms",
                min_value=1,
                max_value=10,
                value=2,
                help="Enter the number of bathrooms"
            )

    with col2:
        st.subheader("Price Estimate")
        if st.button("Calculate Price", key="predict"):
            if bath > bhk + 2:
                st.warning("Having bathrooms more than BHK+2 is unusual. Are you sure?")
            
            if total_sqft/bhk < 200:
                st.warning("The square feet per bedroom seems to be very low. Please verify the input.")
                
            # Make prediction
            price = predict_price(location, total_sqft, bath, bhk, model)
            
            if price is not None:
                # Display prediction in a nice format
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style='text-align: center;'>Estimated Price</h3>
                    <h2 style='text-align: center; color: #ff4b4b;'>₹ {price:.2f} Lakhs</h2>
                    <p style='text-align: center;'>(₹ {price * 0.1:.2f} Million)</p>
                </div>
                """, unsafe_allow_html=True)

    # Additional Information Section
    st.markdown("---")
    st.subheader("📊 Price Insights")
    
    # Create three columns for metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    if 'price' in locals():
        with metric_col1:
            st.metric("Price per Sq.ft", f"₹ {price * 100000 / total_sqft:.2f}", help="Price per square feet")
        with metric_col2:
            st.metric("Price per BHK", f"₹ {price / bhk:.2f} Lakhs", help="Price per bedroom")
        with metric_col3:
            st.metric("Area per BHK", f"{total_sqft / bhk:.2f} sq.ft", help="Square feet per bedroom")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Data source: Bengaluru House Price Dataset</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
