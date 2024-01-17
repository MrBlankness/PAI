import os
import pandas as pd

from .utils import normalize_dataframe, forward_fill_pipeline


data_dir = r'/home/zch/MrLiao/promptEHR/data/challenge2012/'
basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Survival', 'SOFA', 'SAPS-I']
demographic_features = ['Sex', 'Age', 'Height', 'ICUType', 'Weight'] # Sex and ICUType are binary features, others are continuous features
labtest_features = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 
                    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 
                    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 
                    'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 
                    'NISysABP', 'Na', 'PaCO2', 'PaO2', 
                    'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 
                    'TroponinI', 'TroponinT', 'Urine', 'WBC', 'pH']
labtest_features_ori = ['Alkaline phosphatase', 'Alanine transaminase', 'Aspartate transaminase', 'Albumin', 'Blood urea nitrogen', 'Bilirubin',
                        'Serum creatinine', 'Invasive diastolic arterial blood pressure', 'Fractional inspired O2', 'Glasgow Coma Score', 'Serum glucose',
                        'Serum bicarbonate', 'Hematocrit', 'Heart rate', 'Serum potassium', 'Lactate', 'Invasive mean arterial blood pressure',
                        'Mechanical ventilation respiration', 'Serum magnesium', 'Non-invasive diastolic arterial blood pressure', 'Non-invasive mean arterial blood pressure',
                        'Non-invasive systolic arterial blood pressure', 'Serum sodium', 'partial pressure of arterial CO2', 'Partial pressure of arterial O2',
                        'Platelets', 'Respiration rate', 'O2 saturation in hemoglobin', 'Invasive systolic arterial blood pressure', 'Temperature',
                        'Troponin-I', 'Troponin-T', 'Urine output', 'White blood cell count', 'Arterial pH'
                        ]
labtest_features_ori_danwei = ['IU/L', 'IU/L', 'IU/L', 'g/dL', 'mg/dL', 'mg/dL',
                               'mg/dL', 'mmHg', '', '', 'mg/dL',
                               'mmol/L', '%', 'bpm', 'mEq/L', 'mmol/L', 'mmHg',
                               '', 'mmol/L', 'mmHg', 'mmHg', 
                               'mmHg', 'mEq/L', 'mmHg', 'mmHg',
                               'cells/nL', 'bmp', '%', 'mmHg', '°C',
                               'μg/L', 'μg/L', 'mL', 'cells/nL', ''
                               ]
assert len(labtest_features) == len(labtest_features_ori) == len(labtest_features_ori_danwei)
require_impute_features = ['Height', 'Weight'] + labtest_features
normalize_features = ['Age', 'Height', 'Weight'] + labtest_features + ['LOS']

train_df = pd.read_csv(os.path.join(data_dir, "processed", f"train_raw.csv"))
val_df = pd.read_csv(os.path.join(data_dir, "processed", f"val_raw.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "processed", f"test_raw.csv"))

train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

train_x, train_x_mask, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_x, val_x_mask, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_x, test_x_mask, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

for feature in demographic_features + labtest_features:
    if feature not in normalize_features:
        default_fill[feature] = -1