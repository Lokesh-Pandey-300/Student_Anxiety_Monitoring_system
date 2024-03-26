import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import joblib  # Use joblib to save and load models
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut

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
        r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\adityaps.moderateanxiety.csv",
        r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\adityarelax.normal.csv",
        r"C:\Users\hp\OneDrive\Desktop\Labelled datasets\adityascary.normal.csv",
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

    # ... (your file paths here)
]

# Counter for emotion occurrences
emotion_counts = {
    'Moderate Anxiety': 0,
    'Severe Anxiety': 0,
    'Light Anxiety': 0,
    'Normal': 0
}

# Counter for total CSV files
total_csv_files = 0

# Dictionary to store trained model
trained_model = None

# Lists to store data for all emotions
all_X = []
all_y = []

try:
    for excel_file_path in excel_file_paths:
        # Increment the total CSV files count
        total_csv_files += 1
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
                emotion_label = 'Severe Anxiety'
            elif "moderateanxiety" in excel_file_path.lower():
                emotion_label = 'Moderate Anxiety'
            elif "lightanxiety" in excel_file_path.lower():
                emotion_label = 'Light Anxiety'
            elif "normal" in excel_file_path.lower():
                emotion_label = 'Normal'
            # Update the counter for emotion occurrences
            emotion_counts[emotion_label] += 1

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
# Print the counts of each emotion
print("Emotion Occurrences:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")

# Continue with the rest of your code...

# Data distribution column chart
emotion_labels = list(emotion_counts.keys())
emotion_values = list(emotion_counts.values())

plt.bar(emotion_labels, emotion_values)
plt.xlabel('Emotion')
plt.ylabel('Number of Sessions')
plt.title('Data Distribution by Emotion')
plt.show()

# Initialize lists to store training and testing loss, and accuracy
train_loss = []
test_loss = []
train_accuracy_list = []
test_accuracy_list = []

# Check if X_all is empty before attempting to split
if not X_all.empty:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42)

    # Label encoding for the target variable
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize classifiers
    random_forest_classifier = RandomForestClassifier(random_state=42)
    decision_tree_classifier = DecisionTreeClassifier(random_state=42)
    logistic_regression_classifier = LogisticRegression(max_iter=1000, random_state=42)
    svm_classifier = SVC(random_state=42)
    knn_classifier = KNeighborsClassifier()
    naive_bayes_classifier = GaussianNB()
    xgboost_classifier = XGBClassifier(random_state=42)

    # Voting Classifier combining Decision Trees, Random Forest, and Support Vector Machine
    voting_classifier1 = VotingClassifier(
        estimators=[
            ('Random Forest', random_forest_classifier),
            ('Decision Tree', decision_tree_classifier)
        ],
        voting='hard'  # 'hard' for majority voting, 'soft' for weighted voting
    )
    # Voting Classifier combining Decision Trees, Random Forest, and Support Vector Machine
    voting_classifier2 = VotingClassifier(
        estimators=[
            ('Support Vector Machine', svm_classifier),
            ('Decision Tree', decision_tree_classifier)
        ],
        voting='hard'  # 'hard' for majority voting, 'soft' for weighted voting
    )
    voting_classifier3 = VotingClassifier(
        estimators=[
            ('Support Vector Machine', svm_classifier),
            ('Random Forest', random_forest_classifier)
        ],
        voting='hard'  # 'hard' for majority voting, 'soft' for weighted voting
    )
    voting_classifier4 = VotingClassifier(
        estimators=[
            ('Support Vector Machine', svm_classifier),
            ('Random Forest', random_forest_classifier),
            ('Decision Tree', decision_tree_classifier)
        ],
        voting='hard'  # 'hard' for majority voting, 'soft' for weighted voting
    )
    voting_classifier5 = VotingClassifier(
        estimators=[
            ('Support Vector Machine', svm_classifier),
            ('Random Forest', random_forest_classifier),
            ('Decision Tree', decision_tree_classifier),
            ('XGBoost', xgboost_classifier),
        ],
        voting='hard'  # 'hard' for majority voting, 'soft' for weighted voting
    )

    classifiers = {
        'Random Forest': random_forest_classifier,
        'Decision Tree': decision_tree_classifier,
        'Logistic Regression': logistic_regression_classifier,
        'Support Vector Machine': svm_classifier,
        'K-Nearest Neighbors': knn_classifier,
        'Naive Bayes': naive_bayes_classifier,
        'XGBoost': xgboost_classifier,
        'Voting Classifier(RF+DT)': voting_classifier1,
        'Voting Classifier(SVC+DT)': voting_classifier2,
        'Voting Classifier(SVC+RF)': voting_classifier3,
        'Voting Classifier(SVC+RF+DT)': voting_classifier4,
        'Voting Classifier(SVC+RF+DT+XGb)': voting_classifier5,
    }

    for classifier_name, classifier in classifiers.items():
        # Train the model
        classifier.fit(X_train, y_train_encoded)

        # Make predictions on the test set
        y_pred_encoded = classifier.predict(X_test)

        # Convert predictions back to original labels
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # Calculate and print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy for {classifier_name}: {accuracy}")

        # Calculate confusion matrix for the test set
        #cm = confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix using seaborn
        #plt.figure(figsize=(8, 6))
        #sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=list(emotion_counts.keys()),
                    #yticklabels=list(emotion_counts.keys()))
        #plt.xlabel('Predicted')
       # plt.ylabel('Actual')
        #plt.title(f'Confusion Matrix for {classifier_name} (Test Set)')
        #plt.show()

        # Save the trained model to a file
        #joblib.dump(classifier, f'trained_model_{classifier_name.replace(" ", "_").lower()}_emotions.joblib')

        # Training and testing loss/accuracy
        train_loss.append(getattr(classifier, 'loss_', None))  # This may be None, modify as needed
        test_loss.append(classifier.score(X_test, y_test_encoded))
        train_accuracy_list.append(accuracy_score(y_train, label_encoder.inverse_transform(classifier.predict(X_train))))
        test_accuracy_list.append(accuracy)
else:
    print("No data available for splitting. Check previous steps.")

# Plotting training and testing accuracy
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size for better visibility

# Bar width for better visualization
bar_width = 0.35

# Plotting bars
train_bars = ax.bar(classifiers.keys(), train_accuracy_list, width=bar_width, label='Training Accuracy')
test_bars = ax.bar([x for x in classifiers.keys()], test_accuracy_list, width=bar_width, label='Testing Accuracy', alpha=0.7)

# Adding labels and title
plt.xlabel('Classifiers', fontsize=14)  # Increase font size for better visibility
plt.ylabel('Accuracy', fontsize=14)  # Increase font size for better visibility
plt.title('Training and Testing Accuracy Over Classifiers', fontsize=16)  # Increase font size for better visibility
plt.legend()

# Rotating classifier names for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)  # Increase font size for better visibility

# Show the plot with tight layout
plt.tight_layout()

# Showing the plot
plt.show()

# Check if X_all is empty before attempting LOOCV
if not X_all.empty:
    # Define the classifier
    classifier = RandomForestClassifier(random_state=42)

    # Label encoding for the target variable
    label_encoder = LabelEncoder()
    y_all_encoded = label_encoder.fit_transform(y_all)

    # Scale the data
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    # Perform leave-one-out cross-validation
    loocv = LeaveOneOut()
    scores = cross_val_score(classifier, X_all_scaled, y_all_encoded, cv=loocv)

    # Print the cross-validated accuracy scores
    print(f"Cross-validated accuracy scores: {scores}")

    # Print the mean accuracy and standard deviation
    print(f"Mean Accuracy: {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")
else:
    print("No data available for cross-validation. Check previous steps.")

