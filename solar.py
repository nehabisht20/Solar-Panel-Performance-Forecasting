import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Solar Panel Performance Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #FF6B35;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #E55A2B;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">☀️ Solar Panel Performance Forecasting</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Generation", "Data Analysis", "Model Training", "Predictions"])

# Data generation functions
def calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.25 * irradiance - 0.05 * humidity + 0.02 * wind_speed + 0.1 * ambient_temp - 0.03 * abs(tilt_angle - 30))

def calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))

def calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.15 * irradiance - 0.1 * humidity + 0.01 * wind_speed + 0.05 * ambient_temp - 0.04 * abs(tilt_angle - 30))

@st.cache_data
def generate_seasonal_data():
    """Generate synthetic solar panel data for all seasons"""
    
    # Feature ranges for each season
    feature_ranges = {
        'summer': {
            'irradiance': (600, 1000),
            'humidity': (10, 50),
            'wind_speed': (0, 5),
            'ambient_temperature': (30, 45),
            'tilt_angle': (10, 40),
        },
        'winter': {
            'irradiance': (300, 700),
            'humidity': (30, 70),
            'wind_speed': (1, 6),
            'ambient_temperature': (5, 20),
            'tilt_angle': (10, 40),
        },
        'monsoon': {
            'irradiance': (100, 600),
            'humidity': (70, 100),
            'wind_speed': (2, 8),
            'ambient_temperature': (20, 35),
            'tilt_angle': (10, 40),
        }
    }
    
    # Month definitions
    months_days = {
        'summer': {'March': 31, 'April': 30, 'May': 31, 'June': 30},
        'winter': {'November': 30, 'December': 31, 'January': 31, 'February': 28},
        'monsoon': {'July': 31, 'August': 31, 'September': 30, 'October': 31}
    }
    
    calc_functions = {
        'summer': calc_kwh_summer,
        'winter': calc_kwh_winter,
        'monsoon': calc_kwh_monsoon
    }
    
    all_data = []
    
    for season in ['summer', 'winter', 'monsoon']:
        for month, days in months_days[season].items():
            for _ in range(days):
                irr = np.random.uniform(*feature_ranges[season]['irradiance'])
                hum = np.random.uniform(*feature_ranges[season]['humidity'])
                wind = np.random.uniform(*feature_ranges[season]['wind_speed'])
                temp = np.random.uniform(*feature_ranges[season]['ambient_temperature'])
                tilt = np.random.uniform(*feature_ranges[season]['tilt_angle'])
                
                kwh = calc_functions[season](irr, hum, wind, temp, tilt)
                
                all_data.append({
                    'irradiance': round(irr, 2),
                    'humidity': round(hum, 2),
                    'wind_speed': round(wind, 2),
                    'ambient_temperature': round(temp, 2),
                    'tilt_angle': round(tilt, 2),
                    'kwh': round(kwh, 2),
                    'season': season,
                    'month': month
                })
    
    return pd.DataFrame(all_data)

# Generate data
df = generate_seasonal_data()

# PAGE 1: Data Generation
if page == "Data Generation":
    st.header("📊 Data Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generated Dataset")
        st.write(f"**Total Records:** {len(df)}")
        
        # Display data with filters
        season_filter = st.selectbox("Filter by Season", ["All"] + list(df['season'].unique()))
        
        if season_filter != "All":
            filtered_df = df[df['season'] == season_filter]
        else:
            filtered_df = df
        
        st.dataframe(filtered_df, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Summary")
        
        # Season counts
        season_counts = df['season'].value_counts()
        fig = px.pie(values=season_counts.values, names=season_counts.index, 
                    title="Data Distribution by Season")
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        st.write("**Basic Statistics:**")
        st.write(df.describe())
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Dataset as CSV",
        data=csv,
        file_name='solar_performance_data.csv',
        mime='text/csv',
    )

# PAGE 2: Data Analysis
elif page == "Data Analysis":
    st.header("📈 Data Analysis")
    
    # Energy output by season
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Output by Season")
        fig = px.box(df, x='season', y='kwh', title="kWh Distribution by Season")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Day-wise Energy Output")
        fig = px.bar(df, y='kwh', title="Daily Energy Output")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlation Analysis")
    numeric_cols = ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                   title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions by Season")
    
    feature_options = ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']
    selected_feature = st.selectbox("Select feature to analyze", feature_options)
    
    fig = px.histogram(df, x=selected_feature, color='season', 
                      title=f"{selected_feature.title()} Distribution by Season",
                      marginal="box")
    st.plotly_chart(fig, use_container_width=True)

# PAGE 3: Model Training
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    # Model selection
    model_type = st.selectbox("Select Model Type", ["Linear Regression (kWh Prediction)", "Logistic Regression (Season Classification)"])
    
    if model_type == "Linear Regression (kWh Prediction)":
        st.subheader("Linear Regression for kWh Prediction")
        
        # Feature selection
        features = ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']
        selected_features = st.multiselect("Select features for training", features, default=features)
        
        if selected_features:
            # Train-test split
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.1)
            
            X = df[selected_features]
            y = df['kwh']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model
            if st.button("🚀 Train Model"):
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Store model in session state
                st.session_state['regression_model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['selected_features'] = selected_features
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                
                with col2:
                    st.metric("R² Score", f"{r2:.4f}")
                
                with col3:
                    st.metric("Model Accuracy", f"{r2*100:.2f}%")
                
                # Model coefficients
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': model.coef_
                })
                st.dataframe(coef_df)
                
                # Actual vs Predicted plot
                st.subheader("Actual vs Predicted")
                fig = px.scatter(x=y_test, y=y_pred, 
                               labels={'x': 'Actual kWh', 'y': 'Predicted kWh'},
                               title="Actual vs Predicted kWh")
                
                # Add diagonal line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                       mode='lines', name='Perfect Prediction',
                                       line=dict(color='red', dash='dash')))
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Logistic Regression
        st.subheader("Logistic Regression for Season Classification")
        
        # Feature selection (including kWh for classification)
        features = ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']
        selected_features = st.multiselect("Select features for training", features, default=features)
        
        if selected_features:
            # Train-test split
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.1)
            
            X = df[selected_features]
            y = df['season']
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
            
            # Train model
            if st.button("🚀 Train Model"):
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                
                # Store model in session state
                st.session_state['classification_model'] = model
                st.session_state['label_encoder'] = le
                st.session_state['X_test_class'] = X_test
                st.session_state['y_test_class'] = y_test
                st.session_state['selected_features_class'] = selected_features
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                               labels=dict(x="Predicted", y="Actual"),
                               x=le.classes_, y=le.classes_,
                               title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)

# PAGE 4: Predictions
elif page == "Predictions":
    st.header("🔮 Make Predictions")
    
    prediction_type = st.selectbox("Select Prediction Type", ["kWh Prediction", "Season Classification"])
    
    if prediction_type == "kWh Prediction":
        st.subheader("Predict Solar Panel Energy Output")
        
        # Check if model is trained
        if 'regression_model' in st.session_state:
            model = st.session_state['regression_model']
            features = st.session_state['selected_features']
            
            st.write("Enter the following parameters:")
            
            # Input fields
            col1, col2 = st.columns(2)
            
            with col1:
                irradiance = st.number_input("Irradiance (W/m²)", min_value=0.0, max_value=1200.0, value=600.0)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=30.0)
                wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=3.0)
            
            with col2:
                ambient_temp = st.number_input("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
                tilt_angle = st.number_input("Tilt Angle (degrees)", min_value=0.0, max_value=90.0, value=30.0)
            
            # Create input array
            input_data = []
            if 'irradiance' in features:
                input_data.append(irradiance)
            if 'humidity' in features:
                input_data.append(humidity)
            if 'wind_speed' in features:
                input_data.append(wind_speed)
            if 'ambient_temperature' in features:
                input_data.append(ambient_temp)
            if 'tilt_angle' in features:
                input_data.append(tilt_angle)
            
            if st.button("🔮 Predict kWh"):
                prediction = model.predict([input_data])[0]
                st.success(f"Predicted Energy Output: **{prediction:.2f} kWh**")
                
                # Show input summary
                st.subheader("Input Summary")
                input_df = pd.DataFrame({
                    'Parameter': ['Irradiance', 'Humidity', 'Wind Speed', 'Ambient Temperature', 'Tilt Angle'],
                    'Value': [f"{irradiance} W/m²", f"{humidity}%", f"{wind_speed} m/s", f"{ambient_temp}°C", f"{tilt_angle}°"],
                    'Used in Model': ['irradiance' in features, 'humidity' in features, 'wind_speed' in features, 'ambient_temperature' in features, 'tilt_angle' in features]
                })
                st.dataframe(input_df)
        
        else:
            st.warning("Please train the Linear Regression model first in the Model Training section.")
    
    else:  # Season Classification
        st.subheader("Predict Season from Solar Panel Data")
        
        # Check if model is trained
        if 'classification_model' in st.session_state:
            model = st.session_state['classification_model']
            le = st.session_state['label_encoder']
            features = st.session_state['selected_features_class']
            
            st.write("Enter the following parameters:")
            
            # Input fields
            col1, col2 = st.columns(2)
            
            with col1:
                irradiance = st.number_input("Irradiance (W/m²)", min_value=0.0, max_value=1200.0, value=600.0, key="class_irr")
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=30.0, key="class_hum")
                wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=3.0, key="class_wind")
            
            with col2:
                ambient_temp = st.number_input("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, key="class_temp")
                tilt_angle = st.number_input("Tilt Angle (degrees)", min_value=0.0, max_value=90.0, value=30.0, key="class_tilt")
                kwh = st.number_input("Energy Output (kWh)", min_value=0.0, max_value=300.0, value=50.0, key="class_kwh")
            
            # Create input array
            input_data = []
            if 'irradiance' in features:
                input_data.append(irradiance)
            if 'humidity' in features:
                input_data.append(humidity)
            if 'wind_speed' in features:
                input_data.append(wind_speed)
            if 'ambient_temperature' in features:
                input_data.append(ambient_temp)
            if 'tilt_angle' in features:
                input_data.append(tilt_angle)
            if 'kwh' in features:
                input_data.append(kwh)
            
            if st.button("🔮 Predict Season"):
                prediction = model.predict([input_data])[0]
                probabilities = model.predict_proba([input_data])[0]
                
                predicted_season = le.inverse_transform([prediction])[0]
                
                st.success(f"Predicted Season: **{predicted_season.title()}**")
                
                # Show probabilities
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Season': le.classes_,
                    'Probability': probabilities
                })
                prob_df = prob_df.sort_values('Probability', ascending=False)
                
                fig = px.bar(prob_df, x='Season', y='Probability', 
                           title="Season Prediction Probabilities")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please train the Logistic Regression model first in the Model Training section.")

# Footer
st.markdown("---")
st.markdown("**Solar Panel Performance Forecasting App** | Built with Streamlit")
