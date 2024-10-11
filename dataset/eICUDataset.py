import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
# from utils.tools import forward_fill_pipeline, normalize_dataframe, normalize_df_with_statistics
from .utils import normalize_dataframe, forward_fill_pipeline

data_dir = "/data1/MrLiao/PAI/data/eICU"

SEED = 42


basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS'] # ignore Decompensation and Phenotypings
demographic_features = ['Sex', 'Age'] # Sex and ICUType are binary features, others are continuous features
labtest_features = ['admissionheight', 'admissionweight', 'Heart Rate',
       'MAP (mmHg)', 'Invasive BP Diastolic', 'Invasive BP Systolic',
       'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose',
       'FiO2', 'pH'] # ['GCS Total', 'Eyes', 'Motor', 'Verbal'] are categorical features, others are continuous features, we ignore the categorical features due its sparsity if we one-hot encode them
require_impute_features = labtest_features
normalize_features = ['Age'] + labtest_features + ['LOS']

# df = pd.read_csv(os.path.join(data_dir, f"format_eICU.csv"))

# # if a patient has multiple records, we only use the first 48 items
# # we also discard the patients with less than 48 items

# # Ensure dataframe is sorted by PatientID and RecordTime
# df = df.sort_values(['PatientID', 'RecordTime'])

# # Filter out patients with less than 48 records
# df = df.groupby('PatientID').filter(lambda x: len(x) >= 48)

# # Select the first 48 records for each patient
# df = df.groupby('PatientID').head(48)

# # Group the dataframe by patient ID
# grouped = df.groupby('PatientID')

# # Get the patient IDs and outcomes
# patients = np.array(list(grouped.groups.keys()))
# patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# # Get the train_val/test patient IDs
# train_val_patients, test_patients = train_test_split(patients, test_size=20/100, random_state=SEED, stratify=patients_outcome)

# # Get the train/val patient IDs
# train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
# train_patients, val_patients = train_test_split(train_val_patients, test_size=10/80, random_state=SEED, stratify=train_val_patients_outcome)

# # Create train, val, test, [traincal, calib] dataframes for the current fold
# train_df = df[df['PatientID'].isin(train_patients)]
# val_df = df[df['PatientID'].isin(val_patients)]
# test_df = df[df['PatientID'].isin(test_patients)]

# # Save the train, val, and test dataframes for the current fold to csv files
# train_df.to_csv(os.path.join(data_dir, "train_raw.csv"), index=False)
# val_df.to_csv(os.path.join(data_dir, "val_raw.csv"), index=False)
# test_df.to_csv(os.path.join(data_dir, "test_raw.csv"), index=False)


train_df = pd.read_csv(os.path.join(data_dir, f"train_raw.csv"))
val_df = pd.read_csv(os.path.join(data_dir, f"val_raw.csv"))
test_df = pd.read_csv(os.path.join(data_dir, f"test_raw.csv"))

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