# Olympic Swimming Prediction Model

A machine learning system that predicts which Olympic swimming event a swimmer has the best chance of competing in based on their performance data, biometrics, and demographic information.

## 🏊‍♂️ Project Overview

This project uses a Random Forest classifier to analyze swimmer data and predict their optimal Olympic swimming event. The system considers various factors including:

- **Biometric Data**: Height, weight, BMI, age, gender
- **Performance Data**: Best times across different strokes and distances at various ages
- **Demographic Information**: Ethnicity and race
- **Course Types**: Short Course Yards (SCY), Short Course Meters (SCM), Long Course Meters (LCM)

## 📁 Project Structure

```
├── train_swimming_predictor.py      # Main training script
├── predictionScript.py              # Interactive prediction interface
├── dataEngineering.py               # Data preprocessing and feature engineering
├── youthTimesScraper.py             # Web scraper for swimmer performance data
├── nameListScraper.py               # Scraper for Olympic swimmer names
├── staticdata.py                    # Static data and constants
├── final_swimmer_data.csv           # Processed training dataset
├── best_swimming_predictor_model.pkl # Trained model file
├── best_model_parameters.txt        # Model hyperparameters and performance metrics
├── iterative_imputer.pkl            # Saved imputer for handling missing data
├── X_train_imputed.csv              # Training data after imputation
├── X_test_imputed.csv               # Test data after imputation
└── olympic_swimmers_youth_times.json # Scraped performance data
```

## 🚀 Features

### Data Collection & Processing
- **Web Scraping**: Automated collection of Olympic swimmer performance data from SwimCloud
- **Data Engineering**: Comprehensive preprocessing pipeline for swimmer data
- **Missing Data Handling**: Advanced imputation techniques for incomplete performance records
- **Feature Engineering**: Age-specific performance metrics and biometric calculations

### Machine Learning
- **Random Forest Classifier**: Robust prediction model with hyperparameter optimization
- **Multi-class Classification**: Predicts across 12 Olympic swimming events
- **Balanced Performance**: Handles imbalanced dataset with class weighting
- **Cross-validation**: Comprehensive model evaluation

### Interactive Prediction
- **User-friendly Interface**: Interactive command-line tool for predictions
- **Flexible Input**: Accepts various course types and performance data
- **Real-time Analysis**: Instant predictions with confidence scores

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
Install the required packages:

```bash
pip install pandas numpy scikit-learn joblib playwright beautifulsoup4 asyncio
```

### Environment Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Olympic-Swimming-Prediction-Model
```

2. Install Playwright browsers (for web scraping):
```bash
playwright install
```

3. Set up environment variables (optional):
```bash
cp secrets.env.example secrets.env
# Edit secrets.env with your configuration
```

## 📊 Data Pipeline

### 1. Data Collection
- **Name Scraping**: Extract Olympic swimmer names from official sources
- **Performance Scraping**: Collect youth and current performance times from SwimCloud
- **Biometric Data**: Gather physical characteristics and demographic information

### 2. Data Processing
- **Cleaning**: Remove duplicates, handle missing values, standardize formats
- **Feature Engineering**: Create age-specific performance metrics
- **Imputation**: Fill missing performance data using iterative imputation
- **Encoding**: Convert categorical variables to numerical format

### 3. Model Training
- **Feature Selection**: Identify most predictive variables
- **Hyperparameter Tuning**: Optimize Random Forest parameters using RandomizedSearchCV
- **Cross-validation**: Ensure robust model performance
- **Model Persistence**: Save trained model and preprocessing components

## 🎯 Usage

### Training the Model
```bash
python train_swimming_predictor.py
```

This script will:
- Load and preprocess the training data
- Perform hyperparameter optimization
- Train the final model
- Save the model and performance metrics

### Making Predictions
```bash
python predictionScript.py
```

The interactive interface will prompt you for:
1. **Basic Information**: Height, weight, age, gender, ethnicity
2. **Performance Data**: Best times in various strokes and distances
3. **Course Type**: SCY, SCM, or LCM

### Example Prediction Session
```
=== SWIMMING OLYMPIC EVENT PREDICTOR ===
Let's find out which Olympic swimming event you have the best chance in!

Enter your height in cm: 180
Enter your weight in kg: 75
Enter your race/ethnicity: White
Enter your gender (M/F): M
Enter your age: 18

=== COURSE TYPE ===
What course type are your swim times from?
1. SCY (Short Course Yards - 25 yards)
2. SCM (Short Course Meters - 25 meters)
3. LCM (Long Course Meters - 50 meters)

Enter your choice (1/2/3): 1

=== PERFORMANCE DATA ===
Enter your best 50m Freestyle time (seconds): 22.5
...

🎯 PREDICTION RESULTS:
Best Olympic Event: 100m_Freestyle
Confidence: 85.2%
```

### Model Parameters
- **Algorithm**: Random Forest Classifier
- **Estimators**: 200 trees
- **Max Depth**: 40
- **Min Samples Split**: 10
- **Min Samples Leaf**: 2
- **Max Features**: sqrt
- **Class Weight**: balanced
- **Bootstrap**: False

## 🔧 Customization

### Adding New Features
1. Modify `dataEngineering.py` to include new feature calculations
2. Update the training script to incorporate new variables
3. Retrain the model with the enhanced feature set

### Modifying Prediction Events
1. Edit the target event list in `train_swimming_predictor.py`
2. Update the prediction interface in `predictionScript.py`
3. Retrain the model with the new event categories

### Data Source Integration
1. Extend `youthTimesScraper.py` for new data sources
2. Modify data parsing functions in `dataEngineering.py`
3. Update the data pipeline to handle new formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SwimCloud**: Data source for swimmer performance records
- **Olympic Committee**: Official swimmer information
- **Scikit-learn**: Machine learning framework
- **Playwright**: Web scraping automation

## 📞 Support

For questions, issues, or contributions, please:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Contact the development team

---

**Note**: This model is for educational and research purposes. Predictions should not be used as the sole basis for training decisions or Olympic qualification strategies. 
