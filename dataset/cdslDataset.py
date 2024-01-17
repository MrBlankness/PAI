import os
import pandas as pd

from .utils import normalize_dataframe, forward_fill_pipeline

data_dir = r'./data/cdsl/'

basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS']
demographic_features = ['Sex', 'Age']
labtest_features = ['MAX_BLOOD_PRESSURE', 'MIN_BLOOD_PRESSURE', 'TEMPERATURE', 'HEART_RATE', 'OXYGEN_SATURATION', 'ADW -- Coeficiente de anisocitosis', 'ADW -- SISTEMATICO DE SANGRE', 'ALB -- ALBUMINA', 'AMI -- AMILASA', 'AP -- ACTIVIDAD DE PROTROMBINA', 'APTT -- TIEMPO DE CEFALINA (APTT)', 'AU -- ACIDO URICO', 'BAS -- Bas¢filos', 'BAS -- SISTEMATICO DE SANGRE', 'BAS% -- Bas¢filos %', 'BAS% -- SISTEMATICO DE SANGRE', 'BD -- BILIRRUBINA DIRECTA', 'BE(b) -- BE(b)', 'BE(b)V -- BE (b)', 'BEecf -- BEecf', 'BEecfV -- BEecf', 'BT -- BILIRRUBINA TOTAL', 'BT -- BILIRRUBINA TOTAL                                                               ', 'CA -- CALCIO                                                                          ', 'CA++ -- Ca++ Gasometria', 'CHCM -- Conc. Hemoglobina Corpuscular Media', 'CHCM -- SISTEMATICO DE SANGRE', 'CK -- CK (CREATINQUINASA)', 'CL -- CLORO', 'CREA -- CREATININA', 'DD -- DIMERO D', 'EOS -- Eosin¢filos', 'EOS -- SISTEMATICO DE SANGRE', 'EOS% -- Eosin¢filos %', 'EOS% -- SISTEMATICO DE SANGRE', 'FA -- FOSFATASA ALCALINA', 'FER -- FERRITINA', 'FIB -- FIBRINàGENO', 'FOS -- FOSFORO', 'G-CORONAV (RT-PCR) -- Tipo de muestra: Exudado Far¡ngeo/Nasofar¡ngeo', 'GGT -- GGT (GAMMA GLUTAMIL TRANSPEPTIDASA)', 'GLU -- GLUCOSA', 'GOT -- GOT (AST)', 'GPT -- GPT (ALT)', 'HCM -- Hemoglobina Corpuscular Media', 'HCM -- SISTEMATICO DE SANGRE', 'HCO3 -- HCO3-', 'HCO3V -- HCO3-', 'HCTO -- Hematocrito', 'HCTO -- SISTEMATICO DE SANGRE', 'HEM -- Hemat¡es', 'HEM -- SISTEMATICO DE SANGRE', 'HGB -- Hemoglobina', 'HGB -- SISTEMATICO DE SANGRE', 'INR -- INR', 'K -- POTASIO', 'LAC -- LACTATO', 'LDH -- LDH', 'LEUC -- Leucocitos', 'LEUC -- SISTEMATICO DE SANGRE', 'LIN -- Linfocitos', 'LIN -- SISTEMATICO DE SANGRE', 'LIN% -- Linfocitos %', 'LIN% -- SISTEMATICO DE SANGRE', 'MG -- MAGNESIO', 'MONO -- Monocitos', 'MONO -- SISTEMATICO DE SANGRE', 'MONO% -- Monocitos %', 'MONO% -- SISTEMATICO DE SANGRE', 'NA -- SODIO', 'NEU -- Neutr¢filos', 'NEU -- SISTEMATICO DE SANGRE', 'NEU% -- Neutr¢filos %', 'NEU% -- SISTEMATICO DE SANGRE', 'PCO2 -- pCO2', 'PCO2V -- pCO2', 'PCR -- PROTEINA C REACTIVA', 'PH -- pH', 'PHV -- pH', 'PLAQ -- Recuento de plaquetas', 'PLAQ -- SISTEMATICO DE SANGRE', 'PO2 -- pO2', 'PO2V -- pO2', 'PROCAL -- PROCALCITONINA', 'PT -- PROTEINAS TOTALES', 'SO2C -- sO2c (Saturaci¢n de ox¡geno)', 'SO2CV -- sO2c (Saturaci¢n de ox¡geno)', 'TCO2 -- tCO2(B)c', 'TCO2V -- tCO2 (B)', 'TP -- TIEMPO DE PROTROMBINA', 'TROPO -- TROPONINA', 'U -- UREA', 'VCM -- SISTEMATICO DE SANGRE', 'VCM -- Volumen Corpuscular Medio', 'VPM -- SISTEMATICO DE SANGRE', 'VPM -- Volumen plaquetar medio', 'VSG -- VSG']
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
val_x, val_x_mask,  val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_x, test_x_mask, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

for feature in demographic_features + labtest_features:
    if feature not in normalize_features:
        default_fill[feature] = -1