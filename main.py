import pandas as pd
import joblib
import os
from income_data_analyzer import IncomeDataAnalyzer

class Main:
    def __init__(self, data_path, preprocessed_file='preprocessed.csv', preprocessor_file='preprocessor.pkl'):
        self.data_path = data_path
        self.preprocessed_file = preprocessed_file
        self.preprocessor_file = preprocessor_file
        self.analyzer = IncomeDataAnalyzer(data_path)

    def preprocess_and_save(self):
        """Preprocess data and save preprocessed data and preprocessor."""
        # Load and preprocess the data
        try:
            preprocessed_df = self.analyzer.load_and_preprocess(is_training=True)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.preprocessed_file) or '.', exist_ok=True)
            
            # Save the preprocessed data to CSV
            preprocessed_df.to_csv(self.preprocessed_file, index=False)
            print(f"Preprocessed data saved to {self.preprocessed_file}")
            
            # Save the preprocessor
            os.makedirs(os.path.dirname(self.preprocessor_file) or '.', exist_ok=True)
            self.analyzer.save_preprocessor(self.preprocessor_file)
            print(f"Preprocessor saved to {self.preprocessor_file}")
            
            return preprocessed_df
        
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise

    def load_and_predict(self, new_data):
        """Load preprocessor and predict new data."""
        try:
            # Load preprocessor
            self.analyzer.load_preprocessor(self.preprocessor_file)
            
            # Ensure new_data is a DataFrame
            if not isinstance(new_data, pd.DataFrame):
                raise ValueError("new_data must be a pandas DataFrame")
            
            # Predict new data
            predicted_data = self.analyzer.predict_new_data(new_data)
            return predicted_data
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def generate_visualizations_and_report(self):
        """Generate visualizations and report."""
        try:
            # Create visualizations
            report = self.analyzer.create_visualizations()
            print("Visualizations and report generated.")
            return report
        
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            raise

def load_sample_data(file_path):
    """Load and return a sample DataFrame for testing."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return None

if __name__ == '__main__':
    # Define the file paths for data and output
    data_path = 'incomeData.csv'  # Replace with your data file path
    preprocessed_file = 'output/preprocessed.csv'
    preprocessor_file = 'output/preprocessor.pkl'

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Initialize the main class
    main = Main(data_path, preprocessed_file, preprocessor_file)

    try:
        # Step 1: Preprocess data and save it
        preprocessed_df = main.preprocess_and_save()
        
        # Step 2: (Optional) Load new data for prediction
        sample_data = load_sample_data(data_path)
        if sample_data is not None:
            # Take first few rows as new data for prediction
            new_data = sample_data.head(10).drop('Income', axis=1)
            predictions = main.load_and_predict(new_data)
            print("Predicted Data:")
            print(predictions)

        # Step 3: Generate visualizations and report
        report = main.generate_visualizations_and_report()
        print("\nAnalysis Report:")
        print(report)

    except Exception as e:
        print(f"An error occurred: {e}")