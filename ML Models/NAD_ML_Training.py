import pandas as pd

#read the dataset 
kdd_dataset = pd.read_csv("kddcup_dataset.csv", index_col=None)
#examine the proportion of types of traffic
y = kdd_dataset["label"].values
from collections import Counter
Counter(y).most_common()

#convert all non-normal observations into a single class:
def categorize_threats(text):
    """Binarize target threats into normal or system_anomaly."""
    if text == "normal":
        return 0
    else:
        return 1
kdd_dataset["label"] = kdd_dataset["label"].apply(categorize_threats)
Counter(kdd_dataset["label"]).most_common()

# A ratio of anomalies and normal ... later be used in isolation forest
#the values closer to 1 are considered Anomalous and the values that are <0.5 are considered to be "normal".
y = kdd_dataset["label"].values
counts = Counter(y).most_common()
contamination_parameter = counts[1][1] / (counts[0][1] + counts[1][1])

# need to convert all categorical features into numerical state for easy classification & detection
from sklearn.preprocessing import LabelEncoder
encode_dic = dict()
for i in kdd_dataset.columns:
    if kdd_dataset[i].dtype == "object":
        encode_dic[i] = LabelEncoder()
        kdd_dataset[i] = encode_dic[i].fit_transform(kdd_dataset[i])
        
#Splitting the dataset into threat and normal state
kdd_dataset_normal = kdd_dataset[kdd_dataset["label"] == 0]
kdd_dataset_abnormal = kdd_dataset[kdd_dataset["label"] == 1]
y_normal_traffic = kdd_dataset_normal.pop("label").values
X_normal_traffic = kdd_dataset_normal.values
y_anomaly_traffic = kdd_dataset_abnormal.pop("label").values
X_anomaly_traffic = kdd_dataset_abnormal.values

#newly categorized dataset with 0=normal & 1=abnormal
Counter(y).most_common()
#Plitting the dataset for training and testing
from sklearn.model_selection import train_test_split
X_normal_traffic_train, X_normal_traffic_test, y_normal_traffic_train, y_normal_traffic_test = train_test_split(X_normal_traffic, y_normal_traffic, test_size=0.3, random_state=11)

X_anomaly_traffic_train, X_anomaly_traffic_test, y_anomaly_traffic_train, y_anomaly_traffic_test = train_test_split(X_anomaly_traffic, y_anomaly_traffic, test_size=0.3, random_state=11)

import numpy as np
#training
X_train = np.concatenate((X_normal_traffic_train, X_anomaly_traffic_train))
y_train = np.concatenate((y_normal_traffic_train, y_anomaly_traffic_train))
#testing
X_test = np.concatenate((X_normal_traffic_test, X_anomaly_traffic_test))
y_test = np.concatenate((y_normal_traffic_test, y_anomaly_traffic_test))

#after formatting dataset we can now train model employing the classifier (isolation forest)
from sklearn.ensemble import IsolationForest

isolationTrain = IsolationForest(contamination=contamination_parameter)
isolationTrain.fit(X_train)

#Remeber if the score is closer to 1 are considered Anomalous and the score that are
# < 0.5 are considered to be "normal".

#saving the model
import pickle
pickle.dump(isolationTrain, open('NAD_model.pkl', 'wb'))

#testing savevd model
isolationTrain = pickle.load(open('NAD_model.pkl', 'rb'))
print(isolationTrain.decision_function(X_normal_traffic_train))

#Plotting the various scores for a normal network
#Decisioin function score for Normal-Attack where Higher scores indicate normal points, lower scores indicate anomalies
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 4), dpi=600, facecolor="w", edgecolor="k")
plt.hist(classifierScores_taffic_train_normal, bins=50)
plt.title("Normal Network Activity Observation")
plt.show()
print(classifierScores_taffic_train_normal)
#Plotting the various scores for an abnormal network
#Decisioin function score for Abnormal-Attack
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 4), dpi=600, facecolor="w", edgecolor="k")
plt.hist(classifierScores_taffic_train_anomaly, bins=50)
plt.title("Anomalous Network Activity Observation")
plt.show()
print(classifierScores_taffic_train_anomaly)

#declaring various model scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report
#models accuracy score
y_pred = isolationTrain.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")
#precison score
precision = precision_score(y_test, y_pred)
print(f"Precision Score: {precision:.2f}")
#F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")
#recall Score
recall = recall_score(y_test, y_pred)
print(f"Recall Score: {recall:.2f}")
#plotting Accuracy, Precision, Recall, F1-score
plt.figure(figsize=(15, 7))
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1'], [accuracy, precision, recall, f1], color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('Classification Scores')
plt.ylabel('Score')
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")
import seaborn as sns
#plotting confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False, xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
#ROC_AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC: {roc_auc:.2f}")
# Plotting ROC Curve
plt.subplot(1, 2, 2)
plt.plot(color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

