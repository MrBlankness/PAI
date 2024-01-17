import os
import pandas as pd

from .utils import normalize_dataframe, forward_fill_pipeline


data_dir = r'./data/mimic3_4_locf/mimic-iv'
basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS', 'Readmission']
demographic_features = ['Sex', 'Age'] # Sex and ICUType are binary features, others are continuous features
labtest_features = ['Capillary refill rate->0.0', 'Capillary refill rate->1.0',
        'Glascow coma scale eye opening->To Pain',
        'Glascow coma scale eye opening->3 To speech',
        'Glascow coma scale eye opening->1 No Response',
        'Glascow coma scale eye opening->4 Spontaneously',
        'Glascow coma scale eye opening->None',
        'Glascow coma scale eye opening->To Speech',
        'Glascow coma scale eye opening->Spontaneously',
        'Glascow coma scale eye opening->2 To pain',
        'Glascow coma scale motor response->1 No Response',
        'Glascow coma scale motor response->3 Abnorm flexion',
        'Glascow coma scale motor response->Abnormal extension',
        'Glascow coma scale motor response->No response',
        'Glascow coma scale motor response->4 Flex-withdraws',
        'Glascow coma scale motor response->Localizes Pain',
        'Glascow coma scale motor response->Flex-withdraws',
        'Glascow coma scale motor response->Obeys Commands',
        'Glascow coma scale motor response->Abnormal Flexion',
        'Glascow coma scale motor response->6 Obeys Commands',
        'Glascow coma scale motor response->5 Localizes Pain',
        'Glascow coma scale motor response->2 Abnorm extensn',
        'Glascow coma scale total->11', 'Glascow coma scale total->10',
        'Glascow coma scale total->13', 'Glascow coma scale total->12',
        'Glascow coma scale total->15', 'Glascow coma scale total->14',
        'Glascow coma scale total->3', 'Glascow coma scale total->5',
        'Glascow coma scale total->4', 'Glascow coma scale total->7',
        'Glascow coma scale total->6', 'Glascow coma scale total->9',
        'Glascow coma scale total->8',
        'Glascow coma scale verbal response->1 No Response',
        'Glascow coma scale verbal response->No Response',
        'Glascow coma scale verbal response->Confused',
        'Glascow coma scale verbal response->Inappropriate Words',
        'Glascow coma scale verbal response->Oriented',
        'Glascow coma scale verbal response->No Response-ETT',
        'Glascow coma scale verbal response->5 Oriented',
        'Glascow coma scale verbal response->Incomprehensible sounds',
        'Glascow coma scale verbal response->1.0 ET/Trach',
        'Glascow coma scale verbal response->4 Confused',
        'Glascow coma scale verbal response->2 Incomp sounds',
        'Glascow coma scale verbal response->3 Inapprop words',
        'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose',
        'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation',
        'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight',
        'pH']
require_impute_features = labtest_features
normalize_features = ['Age'] + ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose',
        'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation',
        'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight',
        'pH'] + ['LOS']

train_df = pd.read_csv(os.path.join(data_dir, "processed", "ff", f"train_raw.csv"))
val_df = pd.read_csv(os.path.join(data_dir, "processed", "ff", f"val_raw.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "processed", "ff", f"test_raw.csv"))

# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_x, train_x_mask, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_x, val_x_mask, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_x, test_x_mask, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

for feature in demographic_features + labtest_features:
    if feature not in normalize_features:
        default_fill[feature] = -1