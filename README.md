# 🎓 Student Performance Analytics Dashboard

An advanced interactive dashboard for analyzing student performance factors using machine learning and statistical analysis. This Streamlit application provides comprehensive insights into the relationships between study habits, environmental factors, and academic outcomes.

## 🌟 Features

### 📊 Advanced Analytics
- **Polynomial Regression Analysis**: Compare linear vs polynomial models to identify non-linear patterns
- **Feature Combination Experiments**: Test different feature sets to optimize prediction accuracy
- **Multiple Model Comparison**: Linear Regression, Polynomial Regression, and Random Forest models
- **Interactive Visualizations**: Dynamic plots with Plotly for better data exploration

### 🔍 Key Analysis Sections

#### 1. **Overview Dashboard**
- Key Performance Indicators (KPIs)
- Dataset statistics and summary metrics
- Quick data quality assessment

#### 2. **Advanced Data Analysis**
- **Basic Analysis**: Distribution plots and scatter analysis
- **Polynomial Regression**: Compare models from degree 1-5 with cross-validation
- **Feature Experiments**: Test different feature combinations and modeling approaches
- **Correlation Analysis**: Interactive correlation heatmaps
- **Categorical Analysis**: Box plots for categorical factors

#### 3. **AI-Powered Predictions**
- Interactive score predictor with visual feedback
- Real-time predictions based on study hours
- Performance category classification
- Model performance visualization

#### 4. **Insights & Recommendations**
- Key statistical findings
- Actionable recommendations for students
- Prediction tables for different study scenarios

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   git clone <repository-url>
   cd student-performance-analytics
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the dataset**
   - Place `StudentPerformanceFactors.csv` in the project directory
   - The dataset should contain student performance data with various factors

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your web browser at `http://localhost:8501`

## 📁 Project Structure

```
├── app.py                          # Main Streamlit dashboard application
├── requirements.txt                # Python package dependencies
├── README.md                      # This file
└── StudentPerformanceFactors.csv  # Dataset file
```

## 📊 Dataset Requirements

The dashboard expects a CSV file with the following columns:
- `Hours_Studied`: Number of study hours per week
- `Exam_Score`: Target variable (exam scores)
- `Attendance`: Attendance percentage
- `Sleep_Hours`: Hours of sleep per night
- `Motivation_Level`: Student motivation level
- `Parental_Involvement`: Level of parental involvement
- `Access_to_Resources`: Access to educational resources
- And other categorical and numerical features

## 🔧 Key Technologies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **Seaborn & Matplotlib**: Statistical plotting

## 🎯 Model Performance Features

### Polynomial Regression Analysis
- Compare polynomial degrees 1-5
- Cross-validation scoring
- Overfitting detection
- Performance visualization

### Feature Combination Testing
Predefined feature sets:
- **Basic**: Hours studied only
- **Study Focus**: Study hours, attendance, tutoring
- **Personal Factors**: Study hours, sleep, motivation, physical activity
- **Family Support**: Study hours, parental involvement, family income, education
- **School Environment**: Study hours, teacher quality, school type, peer influence
- **Resource Access**: Study hours, resources, internet, distance
- **Comprehensive**: Multiple key factors combined

### Model Comparison
- Linear Regression
- Polynomial Regression (degree 2)
- Random Forest Regression

## 📈 Usage Tips

1. **Start with Overview**: Get familiar with your dataset
2. **Explore Basic Analysis**: Understand data distributions
3. **Try Polynomial Regression**: See if non-linear patterns exist
4. **Experiment with Features**: Find the best feature combinations
5. **Use Predictions**: Test different scenarios
6. **Review Insights**: Get actionable recommendations

## 🔍 Advanced Features

### Custom Feature Analysis
- Select your own feature combinations
- Real-time model training and evaluation
- Feature importance visualization

### Interactive Predictions
- Slider-based input for study hours
- Color-coded performance categories
- Improvement suggestions

### Performance Metrics
- R² Score (coefficient of determination)
- RMSE (Root Mean Square Error)
- Cross-validation scores
- Overfitting detection

## 📱 Dashboard Navigation

The sidebar provides easy navigation between sections:
- **📊 Overview**: Dataset summary and metrics
- **📈 Analysis**: Advanced analytical tools
- **🔮 Predictions**: AI-powered prediction tools
- **💡 Insights**: Key findings and recommendations

## 🎨 Customization

The dashboard uses a modern, professional design with:
- Responsive layout
- Color-coded metrics
- Interactive charts
- Progress indicators
- Custom CSS styling

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data File Not Found**: Ensure `StudentPerformanceFactors.csv` is in the project directory

3. **Performance Issues**: For large datasets, consider data sampling or caching

### Performance Tips
- Use the sidebar quick stats for rapid insights
- Download clean data for external analysis
- Use custom feature analysis for specific research questions

## 📊 Example Insights

The dashboard can reveal insights such as:
- Optimal study hours for maximum performance
- Impact of sleep on academic performance
- Effectiveness of different feature combinations
- Non-linear relationships in student data

## 🔄 Updates and Maintenance

The dashboard is designed to be easily extensible:
- Add new feature combinations in the `get_feature_combinations()` function
- Modify polynomial degrees in the analysis functions
- Customize visualizations using Plotly

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your data format matches requirements
3. Ensure all dependencies are correctly installed

---

**Built with ❤️ using Streamlit | Powered by Machine Learning | Data-Driven Educational Insights**
