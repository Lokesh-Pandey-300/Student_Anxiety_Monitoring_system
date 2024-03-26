import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib  # Use joblib to save and load models

# Function to process values for a given frequency band
def process_frequency_values(segment, target_band):
    target_column = None

    # Find the target column dynamically
    for col_name in segment.columns:
        if target_band.lower() in col_name.lower():
            target_column = col_name
            break

    if target_column is not None:
        # Extract values from the specified target column
        values = segment[target_column]

        # Check if the values need numeric conversion
        if not pd.api.types.is_numeric_dtype(values):
            # Convert to numeric and then fill NaN with 0
            values = pd.to_numeric(values, errors='coerce').fillna(0)

        print(f"\n{target_column} - '{target_band}' Values:\n{values}")
        return values
    else:
        print(f"No column found for {target_band}")
        return pd.Series([])

# List of file paths
excel_file_paths = [
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\aditya.moderateanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\aditya.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\adityarelax.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\adityaspell.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\ashritpq.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\ashritps.moderateanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\ashritrelax.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\ashritscary.severeanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\ashritspell.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\goutampq.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\goutamps.moderateanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\goutamrelax.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\goutamscary.severeanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\goutamspell.severeanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\lokeshpq1.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\lokeshpq2.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\lokeshps1.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\lokeshps2.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\lokeshspell.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\manjupq.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\manjups.severeanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\manjurelax.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\manjuscary.severeanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\manjuspell.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pradeeppq.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pradeepps.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pradeeprelax.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pradeepscary.moderateanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pradeepspell.severeanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pralhadpq1.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pralhadpq2.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pralhadps1.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pralhadps2.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\pralhadspell.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\rohitps.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\rohitrelax.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\rohitscary.moderateanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\rohitspell.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\yasirpq.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\yasirps.lightanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\yasirrelax.normal.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\yasirscary.moderateanxiety.csv",
    r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\yasirspell.moderateanxiety.csv"
    # Add more paths for other emotion files
]

# Dictionary to store trained model
trained_model = None

# Lists to store data for all emotions
all_X = []
all_y = []

try:
    for excel_file_path in excel_file_paths:
        # Load EEG data from the Excel sheet into a DataFrame
        eeg_data = pd.read_csv(excel_file_path, low_memory=False)

        # Print some information for debugging
        print(f"\nProcessing file: {excel_file_path}")
        print("Length of eeg_data:", len(eeg_data))

        # Check if the DataFrame is not empty
        if not eeg_data.empty:
            # Segment EEG signals
            segment_duration_seconds = 5
            sampling_rate = 128
            samples_per_segment = segment_duration_seconds * sampling_rate

            # Target frequency bands
            target_bands = ['Alpha', 'Theta', 'BetaL', 'BetaH', 'Gamma']

            # Dictionary to store values for each frequency band
            all_frequency_values = {}

            # Process values for each frequency band in each segment
            for target_band in target_bands:
                all_frequency_values[target_band] = pd.Series()
                for idx, segment in enumerate([eeg_data.iloc[j:j + samples_per_segment] for j in range(0, len(eeg_data), samples_per_segment)]):
                    print(f"\nSegment {idx + 1}")
                    values = process_frequency_values(segment, target_band)
                    all_frequency_values[target_band] = pd.concat([all_frequency_values[target_band], values], ignore_index=True)

                # Calculate and print the average of values for the current frequency band
                average_value = all_frequency_values[target_band].mean()
                print(f"\nAverage {target_band} Value across Segments: {average_value}")

            # Prepare data for machine learning
            X = pd.DataFrame({
                'Alpha': all_frequency_values['Alpha'],
                'Theta': all_frequency_values['Theta'],
                'BetaL': all_frequency_values['BetaL'],
                'BetaH': all_frequency_values['BetaH'],
                'Gamma': all_frequency_values['Gamma']
            })

            X = X.dropna()

            # Assign labels based on the emotion
            emotion_label = None
            if "severeanxiety" in excel_file_path.lower():
                emotion_label = 0
            elif "moderateanxiety" in excel_file_path.lower():
                emotion_label = 1
            elif "lightanxiety" in excel_file_path.lower():
                emotion_label = 2
            elif "normal" in excel_file_path.lower():
                emotion_label = 3

            # Print the assigned emotion label for verification
            print(f"Emotion label for {excel_file_path}: {emotion_label}")

            # Create target column with labels
            y = pd.Series([emotion_label] * len(X))

            # Append data for the current emotion to the lists
            all_X.append(pd.DataFrame(X))
            all_y.append(pd.Series(y))

        else:
            print("The DataFrame is empty. Please check the file contents.")

except FileNotFoundError as e:
    print(f"File not found at the specified path: {excel_file_path}")
except pd.errors.EmptyDataError:
    print(f"The file at {excel_file_path} is empty.")
except Exception as e:
    print(f"An error occurred: {e}")

# Concatenate data for all emotions if the lists are not empty
if all_X:
    X_all = pd.concat(all_X, ignore_index=True)
else:
    print("No data to concatenate for features.")
    X_all = pd.DataFrame()  # You can create an empty DataFrame or handle it according to your needs

if all_y:
    y_all = pd.concat(all_y, ignore_index=True)
else:
    print("No data to concatenate for labels.")
    y_all = pd.Series()  # You can create an empty Series or handle it according to your needs

# Check if X_all is empty before attempting to split
if not X_all.empty:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    # Continue with the rest of your code (training, prediction, etc.)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy for all emotions: {accuracy}")
    
    # Store the trained model
    trained_model = classifier
    
    # Save the trained model to a file
    if trained_model:
        joblib.dump(trained_model, 'trained_model_combined_emotions.joblib')

else:
    print("No data available for splitting. Check previous steps.")