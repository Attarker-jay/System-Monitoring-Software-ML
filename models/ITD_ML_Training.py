#Feature engineering for ITD
#Feature Engineering for Insider Threat Detection (ITD) ML Code Snippet
import numpy as np
import pandas as pd
dataset_path = "./r42/"

#reading from specific columns in the dataset
log_types_data = ["device", "email", "file", "logon", "http"]
log_fields_list = [
 ["date", "user", "activity"],
 ["date", "user", "to", "cc", "bcc"],
 ["date", "user", "filename"],
 ["date", "user", "activity"],
 ["date", "user", "url"],
]
#engineering various feautures and encoding them by creating a dictionary attack
new_features = 0
new_feature_map = {}

def create_new_feature(name):
    """New feature add-ons onto dictionary for encoding"""
    if name not in new_feature_map:
        global new_features
        new_feature_map[name] = new_features
        new_features += 1
        
# the newly features will be added using  our dictionary
create_new_feature("Weekday_Logon_Normal")
create_new_feature("Weekday_Logon_After")
create_new_feature("Weekend_Logon")
create_new_feature("Logoff")
create_new_feature("Connect_Normal")
create_new_feature("Connect_After")
create_new_feature("Connect_Weekend")
create_new_feature("Disconnect")
create_new_feature("Email_In")
create_new_feature("Email_Out")
create_new_feature("File_exe")
create_new_feature("File_jpg")
create_new_feature("File_zip")
create_new_feature("File_txt")
create_new_feature("File_doc")
create_new_feature("File_pdf")
create_new_feature("File_other")
create_new_feature("url")

#declaring a function that alerts of a file type copied to a removable media
def employee_file_features(row):
    if row["filename"].endswith(".exe"):
        return new_feature_map["File_exe"]
    if row["filename"].endswith(".jpg"):
        return new_feature_map["File_jpg"]
    if row["filename"].endswith(".zip"):
        return new_feature_map["File_zip"]
    if row["filename"].endswith(".txt"):
        return new_feature_map["File_txt"]
    if row["filename"].endswith(".doc"):
        return new_feature_map["File_doc"]
    if row["filename"].endswith(".pdf"):
        return new_feature_map["File_pdf"]
    else:
        return new_feature_map["File_other"]

#declaring a function to detect employess sending emails ouside the company scope
def employee_email_features(row):
    outsider = False
    if not pd.isnull(row["to"]):
        for address in row["to"].split(";"):
            if not address.endswith("dtaa.com"):
                outsider = True
                
    if not pd.isnull(row["cc"]):
        for address in row["cc"].split(";"):
            if not address.endswith("dtaa.com"):
                outsider = True
                
    if not pd.isnull(row["bcc"]):
        for address in row["bcc"].split(";"):
            if not address.endswith("dtaa.com"):
                outsider = True
                
    if outsider:
        return new_feature_map["Email_Out"]
    else:
        return new_feature_map["Email_In"]
    
#declaring a function to detect whether employee used removable media
## during non-working(Business) hours
def employee_device_features(row):
    if row["activity"] == "Connect":
        if row["date"].weekday() < 5:
            if row["date"].hour >= 8 and row["date"].hour < 17:
                return new_feature_map["Connect_Normal"]
            else:
                return new_feature_map["Connect_After"]
        else:
            return new_feature_map["Connect_Weekend"]
    else:
        return new_feature_map["Disconnect"]    
    
#declaring a function to detect whether employee logged onto company Pc's
#outside non-working(business) hours
def employee_logon_features(row):
    if row["activity"] == "Logon":
        if row["date"].weekday() < 5:
            if row["date"].hour >= 8 and row["date"].hour < 17:
                return new_feature_map["Weekday_Logon_Normal"]
            else:
                return new_feature_map["Weekday_Logon_After"]
        else:
            return new_feature_map["Weekend_Logon"]
    else:
        return new_feature_map["Logoff"]
    
#declaring a function to track employee visited URL's (avoid sensitive data)
def employee_http_features(row):
    return new_feature_map["url"]
#declaring a function to extract timestamp of employee activity(only attack day)
def date_to_day(row):
    attack_day_only = row["date"].date()
    return attack_day_only
#Now we can loop through the dataset containing logs and read newly created
# employee features  into pandas data frame
employee_log_feature_functions = [
    employee_device_features,
    employee_email_features,
    employee_file_features,
    employee_logon_features,
    employee_http_features,
]
#we need to convert the data data to timestamp accessible by pandas
dfs = []
for i in range(len(log_types_data)):
    employee_log_type = log_types_data[i]
    employee_log_fields = log_fields_list[i]
    employee_log_feature_function = employee_log_feature_functions[i]
    df = pd.read_csv(dataset_path + employee_log_type + ".csv", usecols=employee_log_fields,index_col=None)
    date_format = "%m/%d/%Y %H:%M:%S"
    df["date"] = pd.to_datetime(df["date"], format=date_format)
    #Declearing the creation of the new feature defined above
    #Droping all features except user,date, and our new feature
    new_employee_feature = df.apply(employee_log_feature_function, axis=1)
    df["feature"] = new_employee_feature
    keep_cols = ["date", "user", "feature"]
    df = df[keep_cols]
    #date conversion to just a day(attack day)
    df["date"] = df.apply(date_to_day, axis=1)
    dfs.append(df)
    
joint = pd.concat(dfs)
joint = joint.sort_values(by="date")
joint

#indexing of dates as timestamps countes in insider attacks(based on dataset SD)
start_date = joint["date"].iloc[0]
end_date = joint["date"].iloc[-1]
time_horizon = (end_date - start_date).days + 1

def indexed_date(date):
    return (date - start_date).days
#declaring a function for the exact time series for a specific employee
def extract_employee_time_series(employee_name, df):
    return df[df["user"] == employee_name]
#vectorizing employess time-series info.. to fit in the model
def vectorize_employee_time_series(employee_name, df):
    employee_time_series = extract_employee_time_series(employee_name, df)
    x = np.zeros((len(new_feature_map), time_horizon))
    event_date_indices = employee_time_series["date"].apply(indexed_date).to_numpy()
    event_features = employee_time_series["feature"].to_numpy()
    for i in range(len(event_date_indices)):
        x[event_features[i], event_date_indices[i]] += 1
    return x
#declaring function for all employees' features in a vectorized time-series style
def dataset_vectorization(df):
    #featurizing the dataset
    employees = set(df["user"].values)
    X = np.zeros((len(employees), len(new_feature_map), time_horizon))
    y = np.zeros((len(employees)))
    for index, employee in enumerate(employees):
        x = vectorize_employee_time_series(employee, df)
        X[index, :, :] = x
        y[index] = int(employee in potential_threats)
    return X, y
X, y = dataset_vectorization(joint)

#Now we can start training and testing our model by splitting it first
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#reshaping vectorized data
X_train_reshaped = X_train.reshape([X_train.shape[0], X_train.shape[1] * X_train.shape[2]])
X_test_reshaped = X_test.reshape([X_test.shape[0], X_test.shape[1] * X_test.shape[2]])

#now we plit the traingin & testing dataset into threat and non-threats
# 0's = normal & 1's = threats
X_train_normal = X_train_reshaped[y_train == 0, :]
print(X_train_normal.shape)

X_train_threat = X_train_reshaped[y_train == 1, :]
print(X_train_threat.shape)

X_test_normal = X_test_reshaped[y_test == 0, :]
print(X_test_normal.shape)

X_test_threat = X_test_reshaped[y_test == 1, :]
print(X_test_threat.shape)

#Finally we can now implement the isolation-forest model
from sklearn.ensemble import IsolationForest

contamination_parameter = 0.035
isolation_train = IsolationForest( n_estimators=100, max_samples=256, contamination=contamination_parameter)
#Time to fit I.F classifier into the training data
isolation_train.fit(X_train_reshaped)
#saving the model
import pickle
pickle.dump(isolation_train, open('ITD_model.pkl', 'wb'))
#testing savevd model
isolation_train = pickle.load(open('ITD_model.pkl', 'rb'))
print(isolation_train.decision_function(X_train_normal))
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#Plotting for normal employee behaviours (score)
normal_scores = isolation_train.decision_function(X_train_normal)

fig = plt.figure(figsize=(8, 4), dpi=600, facecolor="w", edgecolor="k")
normal = plt.hist(normal_scores, 50, density=True)

plt.xlim((-0.2, 0.2))
plt.xlabel("Insider Threat score")
plt.ylabel("Percentage")
plt.title("Distribution of Insider threat score for non-threat employees")
#Plotting for normal employee behaviours (score)
anomaly_scores = isolation_train.decision_function(X_train_threat)

fig = plt.figure(figsize=(8, 4), dpi=600, facecolor="w", edgecolor="k")
normal = plt.hist(anomaly_scores, 50, density=True)

plt.xlim((-0.2, 0.2))
plt.xlabel("Insider Threat score")
plt.ylabel("Percentage")
plt.title("Distribution of Insider threat score for threat employees")
#declaring cutt-off score on the training data
cutoff = 0.5
from collections import Counter

cutoff_score = isolation_train.decision_function(X_train_reshaped)
print(Counter(y_train[cutoff > cutoff_score]))
#declaring cutt-off score on the testing data
cutoff_score = isolation_train.decision_function(X_test_reshaped)
print(Counter(y_test[cutoff > cutoff_score]))
#models accuracy
#declaring various model scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report

#models evaluation matrix
# Predict the test set
y_pred_test = isolation_train.predict(X_test_reshaped)

# Convert the prediction to 0 and 1 (normal and threat)
# In IsolationForest, -1 means anomaly and 1 means normal
y_pred_test = np.where(y_pred_test == -1, 1, 0)
accuracy = accuracy_score(y_test, y_pred_test)
print(accuracy)
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(conf_matrix)
import seaborn as sns
#plotting confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False, xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
#ROC -AUC
roc_auc = roc_auc_score(y_test, y_pred_test)
print(roc_auc)
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
#classification Report
class_report = classification_report(y_test, y_pred_test)
print(class_report)
