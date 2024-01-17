import os
import pandas as pd

from .utils import normalize_dataframe, forward_fill_pipeline

data_dir = r'/home/zch/MrLiao/promptEHR/data/tjh/'

basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS']
demographic_features = ['Sex', 'Age']
labtest_features = ['Hypersensitive cardiac troponinI', 'hemoglobin', 'Serum chloride', 'Prothrombin time', 'procalcitonin', 'eosinophils(%)', 'Interleukin 2 receptor', 'Alkaline phosphatase', 'albumin', 'basophil(%)', 'Interleukin 10', 'Total bilirubin', 'Platelet count', 'monocytes(%)', 'antithrombin', 'Interleukin 8', 'indirect bilirubin', 'Red blood cell distribution width ', 'neutrophils(%)', 'total protein', 'Quantification of Treponema pallidum antibodies', 'Prothrombin activity', 'HBsAg', 'mean corpuscular volume', 'hematocrit', 'White blood cell count', 'Tumor necrosis factorα', 'mean corpuscular hemoglobin concentration', 'fibrinogen', 'Interleukin 1β', 'Urea', 'lymphocyte count', 'PH value', 'Red blood cell count', 'Eosinophil count', 'Corrected calcium', 'Serum potassium', 'glucose', 'neutrophils count', 'Direct bilirubin', 'Mean platelet volume', 'ferritin', 'RBC distribution width SD', 'Thrombin time', '(%)lymphocyte', 'HCV antibody quantification', 'D-D dimer', 'Total cholesterol', 'aspartate aminotransferase', 'Uric acid', 'HCO3-', 'calcium', 'Amino-terminal brain natriuretic peptide precursor(NT-proBNP)', 'Lactate dehydrogenase', 'platelet large cell ratio ', 'Interleukin 6', 'Fibrin degradation products', 'monocytes count', 'PLT distribution width', 'globulin', 'γ-glutamyl transpeptidase', 'International standard ratio', 'basophil count(#)', 'mean corpuscular hemoglobin ', 'Activation of partial thromboplastin time', 'Hypersensitive c-reactive protein', 'HIV antibody quantification', 'serum sodium', 'thrombocytocrit', 'ESR', 'glutamic-pyruvic transaminase', 'eGFR', 'creatinine']
require_impute_features = labtest_features
normalize_features = ['Age'] + labtest_features + ['LOS']

train_df = pd.read_csv(os.path.join(data_dir, "processed", "fold_0", f"train_raw.csv"))
val_df = pd.read_csv(os.path.join(data_dir, "processed", "fold_0", f"val_raw.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "processed", "fold_0", f"test_raw.csv"))

# Normalize data
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Drop rows if all features are recorded NaN
train_df = train_df.dropna(axis=0, how='all', subset=normalize_features)
val_df = val_df.dropna(axis=0, how='all', subset=normalize_features)
test_df = test_df.dropna(axis=0, how='all', subset=normalize_features)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_x, train_x_mask, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_x, val_x_mask, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_x, test_x_mask, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

for feature in demographic_features + labtest_features:
    if feature not in normalize_features:
        default_fill[feature] = -1