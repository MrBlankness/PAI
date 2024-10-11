import os
import pandas as pd

from .utils import normalize_dataframe, forward_fill_pipeline


data_dir = '/data1/MrLiao/PAI/data/sepsis'
basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS']
demographic_features = ['Sex', 'Age', 'HospAdmTime', 'Unit1', 'Unit2', 'ICULOS'] # Sex, Unit1, Unit2 are binary features, others are continuous features
labtest_features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
require_impute_features = ['Age', 'HospAdmTime', 'Unit1', 'Unit2', 'ICULOS'] + labtest_features # Sex normally does not need imputation
normalize_features = ['Age', 'HospAdmTime', 'ICULOS'] + labtest_features + ['LOS']

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