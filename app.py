import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import re
import joblib

# Set page config
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin-top: 20px;
    }
    .feature-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except ImportError:
        st.error("joblib not installed. Install with: pip install joblib")
        return None
    except Exception as e:
        st.error(f"Error loading model with joblib: {str(e)}")
        return None

# Load CSV data and extract unique values for dropdowns
@st.cache_data
def load_csv_data():
    try:
        df = pd.read_csv('laptop_price.csv')
        return df
    except FileNotFoundError:
        st.error("CSV file 'laptop_price.csv' not found. Please ensure the file is in the same directory.")
        return None

def clean_data(df):
    """Clean the dataset by handling missing values and data types"""
    if df is None:
        return None
    
    df_clean = df.copy()
    
    # Handle missing values
    missing_info = df_clean.isnull().sum()
    if missing_info.sum() > 0:
        st.warning(f"Found {missing_info.sum()} missing values in the dataset. Cleaning...")
        
        # Fill missing values based on column type
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    # Ensure proper data types for numerical columns
    numerical_cols = ['Ram', 'Weight', 'Inches', 'X_res', 'Y_res', 'PPI', 'HDD', 'SSD', 'Flash', 'Hybrid', 'Cpu_Speed_GHz', 'Price_euros']
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            if col != 'Price_euros':  # Don't fill target variable
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

@st.cache_data
def get_unique_values(df):
    """Extract unique values from each column for dropdown options"""
    if df is None:
        return {}, {}
    
    unique_values = {
        'Company': sorted(df['Company'].unique().tolist()),
        'TypeName': sorted(df['TypeName'].unique().tolist()),
        'OpSys': sorted(df['OpSys'].unique().tolist()),
        'Cpu_Brand': sorted(df['Cpu_Brand'].unique().tolist()),
        'Gpu_Brand': sorted(df['Gpu_Brand'].unique().tolist())
    }
    
    # Get numerical ranges for better user experience
    ranges = {
        'Ram': {'min': int(df['Ram'].min()), 'max': int(df['Ram'].max())},
        'Inches': {'min': float(df['Inches'].min()), 'max': float(df['Inches'].max())},
        'Weight': {'min': float(df['Weight'].min()), 'max': float(df['Weight'].max())},
        'Cpu_Speed_GHz': {'min': float(df['Cpu_Speed_GHz'].min()), 'max': float(df['Cpu_Speed_GHz'].max())},
        'HDD': {'min': int(df['HDD'].min()), 'max': int(df['HDD'].max())},
        'SSD': {'min': int(df['SSD'].min()), 'max': int(df['SSD'].max())},
        'Flash': {'min': int(df['Flash'].min()), 'max': int(df['Flash'].max())},
        'Hybrid': {'min': int(df['Hybrid'].min()), 'max': int(df['Hybrid'].max())}
    }
    
    return unique_values, ranges

def create_feature_dataframe(input_data, reference_df):
    """Create and preprocess feature dataframe exactly like training data"""
    
    # Create dataframe from input
    df = pd.DataFrame([input_data])
    
    # Handle categorical encoding exactly like training
    categorical_columns = ['Company', 'TypeName', 'OpSys', 'Cpu_Brand', 'Gpu_Brand']
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Get expected columns from reference data (excluding target)
    reference_encoded = pd.get_dummies(reference_df, columns=categorical_columns, drop_first=True)
    if 'Price_euros' in reference_encoded.columns:
        expected_columns = reference_encoded.drop('Price_euros', axis=1).columns
    else:
        expected_columns = reference_encoded.columns
    
    # Add missing columns with 0 values
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Remove extra columns and reorder to match training data
    df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)
    
    return df_encoded

def safe_predict(model, features, reference_df):
    """Safely make prediction with multiple fallback strategies"""
    
    try:
        # Strategy 1: Direct prediction
        prediction = model.predict(features)
        
        # Check if prediction is reasonable (between 100 and 10000 euros)
        pred_value = prediction[0]
        
        # Handle potential log transformation
        if pred_value < 0:
            # Might be log-transformed, try exp
            pred_value = np.exp(pred_value)
        elif pred_value > 100000:
            # Might need log transformation
            pred_value = np.log(pred_value) if pred_value > 0 else abs(pred_value)
        
        # Final sanity check
        if pred_value > 50000 or pred_value < 10:
            # Use median price from training data as fallback
            median_price = reference_df['Price_euros'].median()
            st.warning(f"Prediction seems unrealistic ({pred_value:.2f}). Using median price as reference.")
            return median_price
        
        return pred_value
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Return median price as fallback
        return reference_df['Price_euros'].median()

def get_price_statistics(df):
    """Get price statistics for validation"""
    if df is None or 'Price_euros' not in df.columns:
        return None
    
    stats = {
        'min': df['Price_euros'].min(),
        'max': df['Price_euros'].max(),
        'mean': df['Price_euros'].mean(),
        'median': df['Price_euros'].median(),
        'std': df['Price_euros'].std()
    }
    
    return stats

# Main app
def main():
    st.markdown('<h1 class="main-header">💻 Laptop Price Prediction</h1>', unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    df_raw = load_csv_data()
    
    if model is None or df_raw is None:
        st.error("Unable to load the model or CSV data. Please check if the required files exist.")
        return
    
    # Clean the data
    df = clean_data(df_raw)
    
    if df is None:
        st.error("Unable to clean the data.")
        return
    
    # Show data statistics
    price_stats = get_price_statistics(df)
    if price_stats:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(f"""
        *📊 Price Range in Dataset:*
        - Min: €{price_stats['min']:.2f}
        - Max: €{price_stats['max']:.2f}
        - Average: €{price_stats['mean']:.2f}
        - Median: €{price_stats['median']:.2f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Get unique values and ranges from CSV
    unique_values, ranges = get_unique_values(df)
    
    st.markdown("### Enter Laptop Specifications")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown("💼 Brand & Type**")
        
        company = st.selectbox(
            "Company",
            options=unique_values['Company'],
            help="Select the laptop manufacturer"
        )
        
        type_name = st.selectbox(
            "Type",
            options=unique_values['TypeName'],
            help="Select the laptop type"
        )
        
        op_sys = st.selectbox(
            "Operating System",
            options=unique_values['OpSys'],
            help="Select the operating system"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown("⚙ Hardware Specifications**")
        
        ram = st.number_input(
            "RAM (GB)",
            min_value=1,
            max_value=128,
            value=8,  # Default to common value
            step=1,
            help="Enter RAM size in GB"
        )
        
        inches = st.number_input(
            "Screen Size (inches)",
            min_value=10.0,
            max_value=20.0,
            value=15.6,  # Default to common value
            step=0.1,
            help="Enter screen size in inches"
        )
        
        weight = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,  # Default to common value
            step=0.1,
            help="Enter laptop weight in kg"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown("🖥 Display & Performance**")
        
        cpu_brand = st.selectbox(
            "CPU Brand",
            options=unique_values['Cpu_Brand'],
            help="Select CPU brand"
        )
        
        cpu_speed = st.number_input(
            "CPU Speed (GHz)",
            min_value=0.5,
            max_value=5.0,
            value=2.5,  # Default to common value
            step=0.1,
            help="Enter CPU speed in GHz"
        )
        
        gpu_brand = st.selectbox(
            "GPU Brand",
            options=unique_values['Gpu_Brand'],
            help="Select GPU brand"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional features
    st.markdown("### Additional Features")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown("📱 Display Features**")
        
        touchscreen = st.checkbox("Touchscreen", value=False)
        ips = st.checkbox("IPS Display", value=False)
        
        x_res = st.number_input("X Resolution", min_value=800, max_value=4000, value=1920, step=1)
        y_res = st.number_input("Y Resolution", min_value=600, max_value=3000, value=1080, step=1)
        
        # Calculate PPI
        if inches > 0:
            ppi = round(((x_res**2 + y_res**2)*0.5 / inches), 2)
        else:
            ppi = 0
        st.write(f"*PPI (calculated): {ppi}*")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown("💾 Storage**")
        
        hdd = st.number_input(
            "HDD (GB)",
            min_value=0,
            max_value=10000,
            value=0,  # Default to 0
            step=1,
            help="Enter HDD storage in GB"
        )
        
        ssd = st.number_input(
            "SSD (GB)",
            min_value=0,
            max_value=10000,
            value=256,  # Default to common value
            step=1,
            help="Enter SSD storage in GB"
        )
        
        flash = st.number_input(
            "Flash Storage (GB)",
            min_value=0,
            max_value=10000,
            value=0,  # Default to 0
            step=1,
            help="Enter Flash storage in GB"
        )
        
        hybrid = st.number_input(
            "Hybrid Storage (GB)",
            min_value=0,
            max_value=10000,
            value=0,  # Default to 0
            step=1,
            help="Enter Hybrid storage in GB"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        
        # Prepare input data
        input_data = {
            'Company': company,
            'TypeName': type_name,
            'Inches': float(inches),
            'Ram': float(ram),
            'OpSys': op_sys,
            'Weight': float(weight),
            'Cpu_Brand': cpu_brand,
            'Cpu_Speed_GHz': float(cpu_speed),
            'Touchscreen': int(touchscreen),
            'IPS': int(ips),
            'X_res': float(x_res),
            'Y_res': float(y_res),
            'PPI': float(ppi),
            'Gpu_Brand': gpu_brand,
            'HDD': float(hdd),
            'SSD': float(ssd),
            'Flash': float(flash),
            'Hybrid': float(hybrid)
        }
        
        try:
            # Create feature dataframe
            features = create_feature_dataframe(input_data, df)
            
            # Make safe prediction
            predicted_price = safe_predict(model, features, df)
            
            # Ensure price is reasonable
            predicted_price = max(50, min(predicted_price, 20000))  # Clamp between 50 and 20000 euros
            
            # Display result
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 💰 Predicted Price: €{predicted_price:.2f}")
            st.markdown(f"*≈ ${predicted_price * 1.1:.2f} USD*")
            st.markdown(f"*≈ ₹{predicted_price * 90:.2f} INR*")  # Approximate conversion
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show confidence indicator
            if price_stats:
                if predicted_price < price_stats['min']:
                    st.warning("⚠ Predicted price is below the minimum in dataset")
                elif predicted_price > price_stats['max']:
                    st.warning("⚠ Predicted price is above the maximum in dataset")
                else:
                    st.success("✅ Predicted price is within expected range")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"*Company:* {company}")
                    st.write(f"*Type:* {type_name}")
                    st.write(f"*OS:* {op_sys}")
                    st.write(f"*RAM:* {ram} GB")
                    st.write(f"*Screen:* {inches}\" ({x_res}x{y_res})")
                    st.write(f"*Weight:* {weight} kg")
                
                with col_b:
                    st.write(f"*CPU:* {cpu_brand} @ {cpu_speed} GHz")
                    st.write(f"*GPU:* {gpu_brand}")
                    st.write(f"*Storage:* HDD: {hdd}GB, SSD: {ssd}GB")
                    st.write(f"*Features:* Touchscreen: {touchscreen}, IPS: {ips}")
                    st.write(f"*PPI:* {ppi}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write("Input data:")
                st.json(input_data)
                st.write("Model type:", type(model)._name_)
                st.write("Features shape:", features.shape if 'features' in locals() else "Not created")
                st.write("Error details:", str(e))
    
    # Display sample data
    with st.expander("📋 Sample Data from Dataset"):
        st.dataframe(df.head())
        st.write(f"Dataset shape: {df.shape}")
        
        # Show data types
        st.write("*Data Types:*")
        st.write(df.dtypes)

if __name__ == "__main__":
    main()