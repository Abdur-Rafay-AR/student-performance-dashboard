import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('StudentPerformanceFactors.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'StudentPerformanceFactors.csv' not found!")
        return None

@st.cache_data
def prepare_data(df):
    """Clean and prepare the data"""
    # Handle missing values
    df_clean = df.dropna()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Convert object columns to string to avoid Arrow conversion issues
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
    
    return df_clean

@st.cache_data
def encode_categorical_features(df):
    """Encode categorical variables for advanced analysis"""
    df_encoded = df.copy()
    
    categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                         'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                         'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level',
                         'Distance_from_Home', 'Gender']
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

@st.cache_data
def get_feature_combinations():
    """Define different feature combinations for experimentation"""
    return {
        'basic': ['Hours_Studied'],
        'study_focus': ['Hours_Studied', 'Attendance', 'Tutoring_Sessions'],
        'personal_factors': ['Hours_Studied', 'Sleep_Hours', 'Motivation_Level', 'Physical_Activity'],
        'family_support': ['Hours_Studied', 'Parental_Involvement', 'Family_Income', 'Parental_Education_Level'],
        'school_environment': ['Hours_Studied', 'Teacher_Quality', 'School_Type', 'Peer_Influence'],
        'resources_access': ['Hours_Studied', 'Access_to_Resources', 'Internet_Access', 'Distance_from_Home'],
        'comprehensive': ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Motivation_Level', 
                        'Parental_Involvement', 'Access_to_Resources', 'Previous_Scores'],
    }

@st.cache_data
def analyze_polynomial_regression(df_encoded):
    """Perform polynomial regression analysis"""
    X = df_encoded[['Hours_Studied']].values
    y = df_encoded['Exam_Score'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    degrees = [1, 2, 3, 4, 5]
    
    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
        
        results[degree] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'overfitting': train_r2 - test_r2,
            'model': model,
            'poly_features': poly_features
        }
    
    return results, X_test, y_test

@st.cache_data
def analyze_feature_combinations(df_encoded):
    """Analyze different feature combinations"""
    feature_combinations = get_feature_combinations()
    y = df_encoded['Exam_Score'].values
    scaler = StandardScaler()
    
    results = {}
    
    for name, features in feature_combinations.items():
        # Check if all features exist in the dataset
        available_features = [f for f in features if f in df_encoded.columns]
        if not available_features:
            continue
            
        X = df_encoded[available_features].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        # Polynomial Regression (degree 2)
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_test_poly = poly_features.transform(X_test_scaled)
        
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)
        poly_pred = poly_model.predict(X_test_poly)
        poly_r2 = r2_score(y_test, poly_pred)
        poly_rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        results[name] = {
            'features': available_features,
            'linear_r2': lr_r2,
            'linear_rmse': lr_rmse,
            'poly_r2': poly_r2,
            'poly_rmse': poly_rmse,
            'rf_r2': rf_r2,
            'rf_rmse': rf_rmse,
            'feature_count': len(available_features)
        }
    
    return results

def create_overview_metrics(df):
    """Create overview metrics with enhanced styling"""
    
    # Add section header
    st.markdown("""
    <div class="section-header">
        <h2>📊 Key Performance Indicators</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="👥 Total Students",
                value=f"{len(df):,}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_hours = df['Hours_Studied'].mean()
            st.metric(
                label="📚 Avg Study Hours",
                value=f"{avg_hours:.1f}",
                delta=f"{avg_hours - 20:.1f} vs target"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_score = df['Exam_Score'].mean()
            st.metric(
                label="📝 Avg Exam Score",
                value=f"{avg_score:.1f}",
                delta=f"{avg_score - 70:.1f} vs target"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            correlation = df['Hours_Studied'].corr(df['Exam_Score'])
            correlation_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"
            st.metric(
                label="🔗 Study-Score Correlation",
                value=f"{correlation:.3f}",
                delta=correlation_strength
            )
            st.markdown('</div>', unsafe_allow_html=True)

def create_distribution_plots(df):
    """Create distribution plots with enhanced layout"""
    
    st.markdown("""
    <div class="section-header">
        <h2>📈 Data Distribution Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📚 Study Hours", "📝 Exam Scores", "📊 Combined Analysis"])
    
    with tab1:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        fig_hours = px.histogram(
            df, x='Hours_Studied', 
            nbins=30, 
            title="📚 Distribution of Study Hours",
            color_discrete_sequence=['#667eea'],
            template="plotly_white"
        )
        fig_hours.add_vline(
            x=df['Hours_Studied'].mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {df['Hours_Studied'].mean():.1f}h"
        )
        fig_hours.update_layout(
            showlegend=False,
            height=500,
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig_hours, use_container_width=True)
        
        # Add statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Mean:** {df['Hours_Studied'].mean():.1f} hours")
        with col2:
            st.info(f"**Median:** {df['Hours_Studied'].median():.1f} hours")
        with col3:
            st.info(f"**Std Dev:** {df['Hours_Studied'].std():.1f} hours")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        fig_scores = px.histogram(
            df, x='Exam_Score', 
            nbins=30, 
            title="📝 Distribution of Exam Scores",
            color_discrete_sequence=['#764ba2'],
            template="plotly_white"
        )
        fig_scores.add_vline(
            x=df['Exam_Score'].mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {df['Exam_Score'].mean():.1f}"
        )
        fig_scores.update_layout(
            showlegend=False,
            height=500,
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Add statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"**Mean:** {df['Exam_Score'].mean():.1f} points")
        with col2:
            st.success(f"**Median:** {df['Exam_Score'].median():.1f} points")
        with col3:
            st.success(f"**Std Dev:** {df['Exam_Score'].std():.1f} points")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        # Box plots side by side
        fig_box = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Study Hours Distribution", "Exam Score Distribution")
        )
        
        fig_box.add_trace(
            go.Box(y=df['Hours_Studied'], name='Study Hours', marker_color='#667eea'),
            row=1, col=1
        )
        
        fig_box.add_trace(
            go.Box(y=df['Exam_Score'], name='Exam Scores', marker_color='#764ba2'),
            row=1, col=2
        )
        
        fig_box.update_layout(
            title="📊 Box Plot Comparison",
            title_x=0.5,
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_scatter_analysis(df):
    """Create scatter plot analysis"""
    st.subheader("📈 Study Hours vs Exam Scores Analysis")
    
    # Create scatter plot with trendline
    fig = px.scatter(
        df, 
        x='Hours_Studied', 
        y='Exam_Score',
        title="Relationship between Study Hours and Exam Scores",
        trendline="ols",
        trendline_color_override="red"
    )
    
    # Add correlation annotation
    correlation = df['Hours_Studied'].corr(df['Exam_Score'])
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Correlation: {correlation:.3f}",
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def create_categorical_analysis(df):
    """Create categorical analysis plots"""
    st.subheader("🔍 Categorical Factors Analysis")
    
    # Select categorical columns that exist in the dataset
    categorical_cols = []
    potential_cols = ['Parental_Involvement', 'Motivation_Level', 'Access_to_Resources', 
                     'Internet_Access', 'School_Type', 'Gender']
    
    for col in potential_cols:
        if col in df.columns:
            categorical_cols.append(col)
    
    if categorical_cols:
        col1, col2 = st.columns(2)
        
        # First categorical variable
        if len(categorical_cols) > 0:
            with col1:
                fig1 = px.box(
                    df, 
                    x=categorical_cols[0], 
                    y='Exam_Score',
                    title=f"Exam Scores by {categorical_cols[0]}",
                    color=categorical_cols[0]
                )
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
        
        # Second categorical variable
        if len(categorical_cols) > 1:
            with col2:
                fig2 = px.box(
                    df, 
                    x=categorical_cols[1], 
                    y='Exam_Score',
                    title=f"Exam Scores by {categorical_cols[1]}",
                    color=categorical_cols[1]
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

def create_polynomial_analysis(df_encoded):
    """Create polynomial regression analysis interface"""
    st.markdown("""
    <div class="section-header">
        <h2>🔄 Polynomial Regression Analysis</h2>
        <p>Explore how polynomial features improve prediction accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Perform polynomial analysis
    poly_results, X_test, y_test = analyze_polynomial_regression(df_encoded)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Performance Comparison", "📈 Prediction Curves", "🎯 Best Model"])
    
    with tab1:
        # Performance metrics comparison
        degrees = list(poly_results.keys())
        train_r2 = [poly_results[d]['train_r2'] for d in degrees]
        test_r2 = [poly_results[d]['test_r2'] for d in degrees]
        cv_mean = [poly_results[d]['cv_mean'] for d in degrees]
        rmse = [poly_results[d]['test_rmse'] for d in degrees]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(x=degrees, y=train_r2, mode='lines+markers', 
                                      name='Training R²', line=dict(color='#1f77b4', width=3)))
            fig_r2.add_trace(go.Scatter(x=degrees, y=test_r2, mode='lines+markers', 
                                      name='Testing R²', line=dict(color='#ff7f0e', width=3)))
            fig_r2.add_trace(go.Scatter(x=degrees, y=cv_mean, mode='lines+markers', 
                                      name='CV R²', line=dict(color='#2ca02c', width=3)))
            fig_r2.update_layout(title="R² Score vs Polynomial Degree", 
                               xaxis_title="Polynomial Degree", yaxis_title="R² Score")
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Scatter(x=degrees, y=rmse, mode='lines+markers', 
                                        name='Test RMSE', line=dict(color='#d62728', width=3)))
            fig_rmse.update_layout(title="RMSE vs Polynomial Degree", 
                                 xaxis_title="Polynomial Degree", yaxis_title="RMSE")
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Performance table
        st.subheader("📋 Detailed Performance Metrics")
        perf_data = []
        for degree in degrees:
            perf_data.append({
                'Degree': degree,
                'Train R²': f"{poly_results[degree]['train_r2']:.4f}",
                'Test R²': f"{poly_results[degree]['test_r2']:.4f}",
                'CV R²': f"{poly_results[degree]['cv_mean']:.4f} ± {poly_results[degree]['cv_std']:.4f}",
                'RMSE': f"{poly_results[degree]['test_rmse']:.4f}",
                'Overfitting': f"{poly_results[degree]['overfitting']:.4f}"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
    
    with tab2:
        # Prediction curves for different degrees
        st.subheader("📈 Prediction Curves Comparison")
        
        degrees_to_show = st.multiselect(
            "Select polynomial degrees to compare:",
            options=degrees,
            default=[1, 2, 3],
            help="Choose which polynomial degrees to visualize"
        )
        
        if degrees_to_show:
            fig_curves = go.Figure()
            
            # Add scatter plot of actual data
            fig_curves.add_trace(go.Scatter(
                x=X_test.flatten(), y=y_test, mode='markers',
                name='Actual Data', marker=dict(color='gray', size=5, opacity=0.6)
            ))
            
            # Add prediction curves
            X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, degree in enumerate(degrees_to_show):
                model = poly_results[degree]['model']
                poly_features = poly_results[degree]['poly_features']
                X_range_poly = poly_features.transform(X_range)
                y_range_pred = model.predict(X_range_poly)
                
                fig_curves.add_trace(go.Scatter(
                    x=X_range.flatten(), y=y_range_pred, mode='lines',
                    name=f'Degree {degree} (R²={poly_results[degree]["test_r2"]:.3f})',
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
            
            fig_curves.update_layout(
                title="Polynomial Regression Curves",
                xaxis_title="Hours Studied",
                yaxis_title="Exam Score",
                height=500
            )
            st.plotly_chart(fig_curves, use_container_width=True)
    
    with tab3:
        # Best model details
        best_degree = max(poly_results.keys(), key=lambda k: poly_results[k]['test_r2'])
        best_model = poly_results[best_degree]
        
        st.subheader(f"🏆 Best Model: Polynomial Degree {best_degree}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test R²", f"{best_model['test_r2']:.4f}")
        with col2:
            st.metric("Test RMSE", f"{best_model['test_rmse']:.4f}")
        with col3:
            st.metric("CV Score", f"{best_model['cv_mean']:.4f}")
        with col4:
            st.metric("Overfitting", f"{best_model['overfitting']:.4f}")
        
        # Model interpretation
        if best_degree == 1:
            st.success("🔍 **Linear model is optimal** - Relationship is predominantly linear")
        elif best_degree == 2:
            st.info("📈 **Quadratic model** - Some non-linear patterns detected")
        elif best_degree >= 3:
            st.warning("⚠️ **Higher-order polynomial** - Check for overfitting")

def create_feature_combination_analysis(df_encoded):
    """Create feature combination analysis interface"""
    st.markdown("""
    <div class="section-header">
        <h2>🧪 Feature Combination Experiments</h2>
        <p>Compare different feature sets and modeling approaches</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Perform feature combination analysis
    feature_results = analyze_feature_combinations(df_encoded)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Overview", "🔍 Feature Impact", "🏆 Best Models", "🎮 Custom Analysis"])
    
    with tab1:
        # Overall performance comparison
        combinations = list(feature_results.keys())
        linear_r2 = [feature_results[c]['linear_r2'] for c in combinations]
        poly_r2 = [feature_results[c]['poly_r2'] for c in combinations]
        rf_r2 = [feature_results[c]['rf_r2'] for c in combinations]
        
        # Bar chart comparison
        fig_comparison = go.Figure(data=[
            go.Bar(name='Linear', x=combinations, y=linear_r2, marker_color='#1f77b4'),
            go.Bar(name='Polynomial', x=combinations, y=poly_r2, marker_color='#ff7f0e'),
            go.Bar(name='Random Forest', x=combinations, y=rf_r2, marker_color='#2ca02c')
        ])
        
        fig_comparison.update_layout(
            title="Model Performance by Feature Combination",
            xaxis_title="Feature Combination",
            yaxis_title="R² Score",
            barmode='group',
            height=500
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Performance table
        st.subheader("📋 Detailed Results")
        results_data = []
        for combo in combinations:
            results_data.append({
                'Feature Set': combo.replace('_', ' ').title(),
                'Features Count': feature_results[combo]['feature_count'],
                'Linear R²': f"{feature_results[combo]['linear_r2']:.4f}",
                'Polynomial R²': f"{feature_results[combo]['poly_r2']:.4f}",
                'Random Forest R²': f"{feature_results[combo]['rf_r2']:.4f}",
                'Best Model': max([
                    ('Linear', feature_results[combo]['linear_r2']),
                    ('Polynomial', feature_results[combo]['poly_r2']),
                    ('Random Forest', feature_results[combo]['rf_r2'])
                ], key=lambda x: x[1])[0]
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        # Feature count vs performance
        feature_counts = [feature_results[c]['feature_count'] for c in combinations]
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=feature_counts, y=linear_r2, mode='markers+text',
            text=combinations, textposition="top center",
            name='Linear', marker=dict(color='#1f77b4', size=10)
        ))
        fig_scatter.add_trace(go.Scatter(
            x=feature_counts, y=poly_r2, mode='markers+text',
            text=combinations, textposition="bottom center",
            name='Polynomial', marker=dict(color='#ff7f0e', size=10)
        ))
        fig_scatter.add_trace(go.Scatter(
            x=feature_counts, y=rf_r2, mode='markers+text',
            text=combinations, textposition="middle right",
            name='Random Forest', marker=dict(color='#2ca02c', size=10)
        ))
        
        fig_scatter.update_layout(
            title="Performance vs Number of Features",
            xaxis_title="Number of Features",
            yaxis_title="R² Score",
            height=500
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Best performing models
        all_scores = []
        for combo in combinations:
            all_scores.append(('Linear-' + combo, feature_results[combo]['linear_r2']))
            all_scores.append(('Polynomial-' + combo, feature_results[combo]['poly_r2']))
            all_scores.append(('RandomForest-' + combo, feature_results[combo]['rf_r2']))
        
        # Sort by performance
        all_scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = all_scores[:5]
        
        st.subheader("🏆 Top 5 Model-Feature Combinations")
        for i, (model_combo, score) in enumerate(top_5, 1):
            model_type, feature_set = model_combo.split('-', 1)
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.write(f"**#{i}**")
            with col2:
                st.write(f"**{model_type}** with {feature_set.replace('_', ' ').title()}")
            with col3:
                st.metric("R²", f"{score:.4f}")
        
        # Best overall recommendation
        best_model, best_score = top_5[0]
        st.success(f"🎯 **Recommended Model**: {best_model.replace('-', ' with ').replace('_', ' ')} (R² = {best_score:.4f})")
    
    with tab4:
        # Custom feature selection
        st.subheader("🎮 Custom Feature Analysis")
        
        available_features = [col for col in df_encoded.columns if col != 'Exam_Score']
        selected_features = st.multiselect(
            "Select features for custom analysis:",
            options=available_features,
            default=['Hours_Studied', 'Sleep_Hours', 'Motivation_Level'],
            help="Choose features to include in your custom model"
        )
        
        if selected_features and st.button("🚀 Run Custom Analysis"):
            # Run custom analysis
            X_custom = df_encoded[selected_features].values
            y_custom = df_encoded['Exam_Score'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test different models
            models = {
                'Linear': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            custom_results = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                custom_results[name] = {'r2': r2, 'rmse': rmse}
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Linear Regression R²", f"{custom_results['Linear']['r2']:.4f}")
                st.metric("Linear Regression RMSE", f"{custom_results['Linear']['rmse']:.4f}")
            with col2:
                st.metric("Random Forest R²", f"{custom_results['Random Forest']['r2']:.4f}")
                st.metric("Random Forest RMSE", f"{custom_results['Random Forest']['rmse']:.4f}")
            
            # Feature importance for Random Forest
            if len(selected_features) > 1:
                rf_model = models['Random Forest']
                importance = rf_model.feature_importances_
                
                fig_importance = go.Figure(data=[
                    go.Bar(x=selected_features, y=importance, marker_color='#2ca02c')
                ])
                fig_importance.update_layout(
                    title="Feature Importance (Random Forest)",
                    xaxis_title="Features",
                    yaxis_title="Importance"
                )
                st.plotly_chart(fig_importance, use_container_width=True)

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    st.subheader("🔗 Feature Correlation Matrix")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Numerical Features"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def build_prediction_model(df):
    """Build and evaluate prediction model"""
    st.subheader("🔮 Prediction Model Analysis")
    
    # Prepare data
    X = df[['Hours_Studied']].values
    y = df['Exam_Score'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.4f}", f"{r2*100:.2f}%")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}", "points")
    with col3:
        st.metric("MAE", f"{mae:.2f}", "points")
    
    # Model equation
    st.info(f"📐 **Model Equation:** Exam Score = {model.intercept_:.2f} + {model.coef_[0]:.4f} × Study Hours")
    
    # Prediction vs Actual plot
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(opacity=0.7)
        ))
        
        # Perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            title=f"Actual vs Predicted Scores (R² = {r2:.3f})",
            xaxis_title="Actual Scores",
            yaxis_title="Predicted Scores"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with col2:
        # Residuals plot
        residuals = y_test - y_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(opacity=0.7)
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(
            title="Residuals Plot",
            xaxis_title="Predicted Scores",
            yaxis_title="Residuals"
        )
        st.plotly_chart(fig_res, use_container_width=True)
    
    return model

def create_prediction_tool(model):
    """Create interactive prediction tool with enhanced styling"""
    
    st.markdown("""
    <div class="section-header">
        <h2>🔮 AI-Powered Score Predictor</h2>
        <p>Use the slider below to predict exam scores based on study hours</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create enhanced layout
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Input Parameters")
        
        hours_input = st.slider(
            "📚 Study Hours per Week:",
            min_value=1,
            max_value=50,
            value=20,
            step=1,
            help="Select the number of hours studied per week"
        )
        
        # Make prediction
        predicted_score = model.predict([[hours_input]])[0]
        
        # Enhanced prediction display
        st.markdown("### 📊 Prediction Result")
        
        # Score with color coding
        if predicted_score >= 80:
            score_color = "#4CAF50"  # Green
            category = "🌟 Excellent"
            emoji = "🎉"
        elif predicted_score >= 70:
            score_color = "#2196F3"  # Blue
            category = "👍 Good"
            emoji = "😊"
        elif predicted_score >= 60:
            score_color = "#FF9800"  # Orange
            category = "⚠️ Average"
            emoji = "😐"
        else:
            score_color = "#F44336"  # Red
            category = "❌ Below Average"
            emoji = "😟"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {score_color}20, {score_color}10);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            border: 2px solid {score_color};
            margin: 1rem 0;
        ">
            <h1 style="color: {score_color}; margin: 0; font-size: 3rem;">{predicted_score:.1f}</h1>
            <p style="margin: 0.5rem 0; font-size: 1.2rem; color: {score_color};">Points</p>
            <h3 style="margin: 0; color: {score_color};">{category} {emoji}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights
        improvement_hours = max(0, (75 - predicted_score) / model.coef_[0])
        if improvement_hours > 0 and improvement_hours < 50:
            st.info(f"💡 **Tip:** Study {improvement_hours:.0f} more hours per week to reach 75 points!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        
        # Create enhanced prediction visualization
        hours_range = np.arange(1, 51)
        predictions_range = model.predict(hours_range.reshape(-1, 1))
        
        fig = go.Figure()
        
        # Prediction curve with gradient
        fig.add_trace(go.Scatter(
            x=hours_range,
            y=predictions_range,
            mode='lines',
            name='Prediction Curve',
            line=dict(color='#667eea', width=4),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        # Current prediction point
        fig.add_trace(go.Scatter(
            x=[hours_input],
            y=[predicted_score],
            mode='markers',
            name='Your Prediction',
            marker=dict(
                color='#764ba2', 
                size=20, 
                symbol='star',
                line=dict(color='white', width=2)
            ),
            hovertemplate=f'<b>Study Hours:</b> {hours_input}<br><b>Predicted Score:</b> {predicted_score:.1f}<extra></extra>'
        ))
        
        # Add target lines
        fig.add_hline(y=70, line_dash="dash", line_color="green", 
                     annotation_text="Good Performance (70+)")
        fig.add_hline(y=80, line_dash="dash", line_color="blue", 
                     annotation_text="Excellent Performance (80+)")
        
        fig.update_layout(
            title="📈 Score Prediction Curve",
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Study Hours per Week",
            yaxis_title="Predicted Exam Score",
            height=500,
            template="plotly_white",
            showlegend=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance zones
        st.markdown("### 🎯 Performance Zones")
        zones_col1, zones_col2 = st.columns(2)
        
        with zones_col1:
            st.success("🌟 **Excellent:** 80+ points")
            st.info("👍 **Good:** 70-79 points")
        
        with zones_col2:
            st.warning("⚠️ **Average:** 60-69 points")
            st.error("❌ **Below Average:** < 60 points")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application with enhanced layout"""
    
    # Header with enhanced styling
    st.markdown("""
    <div class="main-header">
        🎓 Student Performance Analytics Dashboard
    </div>
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <p style="font-size: 1.2rem;">Advanced Analytics & AI-Powered Insights for Academic Success</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with error handling
    with st.spinner("🔄 Loading dataset..."):
        df = load_data()
        if df is None:
            st.stop()
    
    # Enhanced sidebar with better navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: white;">📋 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "Choose a section:",
            ["📊 Overview", "📈 Analysis", "🔮 Predictions", "💡 Insights"],
            help="Select a section to explore different aspects of the data"
        )
        
        # Add data summary in sidebar
        st.markdown("---")
        st.markdown("### 📋 Quick Stats")
        df_clean = prepare_data(df)
        st.metric("Students", f"{len(df_clean):,}")
        st.metric("Features", f"{df_clean.shape[1]}")
        st.metric("Avg Score", f"{df_clean['Exam_Score'].mean():.1f}")
        
        # Add download option
        st.markdown("---")
        if st.button("📥 Download Clean Data"):
            csv = df_clean.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name="clean_student_data.csv",
                mime="text/csv"
            )
    
    # Data preparation with progress
    with st.spinner("🧹 Preparing data..."):
        df_clean = prepare_data(df)
    
    # Display selected page with enhanced styling
    if page == "📊 Overview":
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.header("📊 Dataset Overview")
        
        # Overview metrics
        create_overview_metrics(df_clean)
        
        # Enhanced data info section
        st.markdown("---")
        
        # Data info in tabs
        info_tab1, info_tab2, info_tab3 = st.tabs(["📋 Basic Info", "👀 Sample Data", "📊 Statistics"])
        
        with info_tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Shape:** {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
            with col2:
                st.info(f"**Memory:** {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            with col3:
                st.info(f"**Missing Values:** {df_clean.isnull().sum().sum()}")
        
        with info_tab2:
            st.dataframe(
                df_clean.head(10), 
                use_container_width=True,
                height=400
            )
        
        with info_tab3:
            # Only show numerical columns for statistics
            numerical_df = df_clean.select_dtypes(include=[np.number])
            st.dataframe(
                numerical_df.describe(), 
                use_container_width=True,
                height=400
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "📈 Analysis":
        st.header("📈 Advanced Data Analysis")
        
        # Prepare encoded data for advanced analysis
        with st.spinner("🔄 Encoding categorical features..."):
            df_encoded = encode_categorical_features(df_clean)
        
        # Create analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
            "📊 Basic Analysis", "🔄 Polynomial Regression", "🧪 Feature Experiments", "🔗 Correlations", "📂 Categories"
        ])
        
        with analysis_tab1:
            # Distribution plots
            create_distribution_plots(df_clean)
            
            # Scatter analysis
            create_scatter_analysis(df_clean)
        
        with analysis_tab2:
            # Polynomial regression analysis
            create_polynomial_analysis(df_encoded)
        
        with analysis_tab3:
            # Feature combination analysis
            create_feature_combination_analysis(df_encoded)
        
        with analysis_tab4:
            # Correlation heatmap
            create_correlation_heatmap(df_clean)
        
        with analysis_tab5:
            # Categorical analysis
            create_categorical_analysis(df_clean)
    
    elif page == "🔮 Predictions":
        st.header("🔮 Machine Learning Predictions")
        
        # Build model with progress
        with st.spinner("🔮 Training AI model..."):
            model = build_prediction_model(df_clean)
        
        # Prediction tool
        create_prediction_tool(model)
    
    elif page == "💡 Insights":
        st.markdown('<div class="custom-container">', unsafe_allow_html=True)
        st.header("💡 Key Insights & Recommendations")
        
        # Calculate key statistics
        correlation = df_clean['Hours_Studied'].corr(df_clean['Exam_Score'])
        avg_hours = df_clean['Hours_Studied'].mean()
        avg_score = df_clean['Exam_Score'].mean()
        
        # Build simple model for coefficient
        X = df_clean[['Hours_Studied']].values
        y = df_clean['Exam_Score'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Enhanced insights section
        insights_tab1, insights_tab2, insights_tab3 = st.tabs(["🔍 Key Findings", "🎯 Recommendations", "📋 Predictions"])
        
        with insights_tab1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            insights = [
                f"📚 Students study an average of **{avg_hours:.1f} hours** per week",
                f"📝 Average exam score is **{avg_score:.1f} points**",
                f"🔗 Study hours and exam scores have a **{correlation:.3f}** correlation",
                f"📈 Each additional study hour increases scores by **{model.coef_[0]:.2f} points**",
                f"🎯 Students who don't study are predicted to score **{model.intercept_:.1f} points**"
            ]
            
            for insight in insights:
                st.markdown(f"- {insight}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with insights_tab2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            recommendations = [
                "📖 **Study Consistency:** Maintain regular study schedules rather than cramming",
                "⏰ **Optimal Hours:** Aim for 20-25 hours of study per week for best results",
                "🔍 **Quality Focus:** Consider study methods and environment, not just time",
                "📊 **Multiple Factors:** Remember that study hours alone don't determine success",
                "💪 **Holistic Approach:** Consider sleep, motivation, and resources for better outcomes"
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with insights_tab3:
            # Performance prediction table
            sample_hours = list(range(5, 45, 5))
            predictions_data = []
            
            for hours in sample_hours:
                pred_score = model.predict([[hours]])[0]
                if pred_score >= 80:
                    category = "🌟 Excellent"
                elif pred_score >= 70:
                    category = "👍 Good"
                elif pred_score >= 60:
                    category = "⚠️ Average"
                else:
                    category = "❌ Below Average"
                
                predictions_data.append({
                    "Study Hours": hours,
                    "Predicted Score": f"{pred_score:.1f}",
                    "Performance": category
                })
            
            predictions_df = pd.DataFrame(predictions_data)
            st.dataframe(predictions_df, use_container_width=True, height=400)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>📊 Built with Streamlit | 🔮 Powered by Machine Learning | 📈 Data-Driven Insights</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
