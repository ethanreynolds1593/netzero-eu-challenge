"""Make sure there's a `data` folder at the same level as the `src` folder before running this script. It should contain all the necessary dataset files for the code to work correctly."""

"""Since the CSV file is large, data processing might take some time."""

"""Run `python src/02_data_processing.py`"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Get the directory of the current script
script_dir = os.getcwd()

# Construct the file path dynamically for input dataset
data_folder = os.path.join(script_dir, ".", "data")
file_path = os.path.join(data_folder, "cleaned_data.csv")

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# Load the dataset
df = pd.read_csv(file_path, delimiter=",", low_memory=False, dtype=str)

# Clean column names by stripping leading/trailing spaces
df.columns = df.columns.str.strip()

# Define categorical columns
categorical_columns = ['dwellingtypedescr', 'typeofrating', 'energyrating', 'berrating',
                       'mainspaceheatingfuel', 'mainwaterheatingfuel', 'structuretype', 
                       'insulationtype', 'thermalmasscategory', 'predominantrooftype',
                       'heatsystemcontrolcat', 'heatsystemresponsecat', 'purposeofrating',
                       'firstenergytype_description', 'secondenergytype_description', 'thirdenergytype_description', 
                       'firstwalltype_description']

# Initialize the label encoder
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns with a progress bar
for col in tqdm(categorical_columns, desc="Encoding Categorical Columns"):
    if col in df.columns:  # Check if the column exists in the dataframe
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    else:
        print(f"Column '{col}' not found in the dataset, skipping encoding.")

# Define numerical columns
numerical_columns = ['year_of_construction', 'groundfloorarea(sq m)', 'uvaluewall', 'uvalueroof', 'uvaluefloor',
                     'uvaluewindow', 'uvaluedoor', 'wallarea', 'roofarea', 'floorarea', 'windowarea', 'doorarea',
                     'nostoreys', 'co2rating', 'hsmainsystemefficiency', 'mpcdervalue', 'hseffadjfactor',
                     'hssupplheatfraction', 'hssupplsystemeff', 'whmainsystemeff', 'wheffadjfactor', 'supplshfuel',
                     'supplwhfuel', 'shrenewableresources', 'whrenewableresources', 'noofchimneys', 'noofopenflues',
                     'nooffansandvents', 'nooffluelessgasfires', 'fanpowermanudeclaredvalue', 'heatexchangereff',
                     'suspendedwoodenfloor', 'percentagedraughtstripped', 'noofsidessheltered', 'permeabilitytest',
                     'permeabilitytestresult', 'tempadjustment', 'nocentralheatingpumps', 'chboilerthermostatcontrolled',
                     'nooilboilerheatingpumps', 'obboilerthermostatcontrolled', 'obpumpinsidedwelling',
                     'nogasboilerheatingpumps', 'warmairheatingsystem', 'undergroundheating', 'groundflooruvalue',
                     'distributionlosses', 'storagelosses', 'manulossfactoravail', 'solarhotwaterheating',
                     'elecimmersioninsummer', 'combiboiler', 'keephotfacility', 'waterstoragevolume', 'declaredlossfactor',
                     'tempfactorunadj', 'tempfactormultiplier', 'insulationthickness', 'primarycircuitloss',
                     'combiboileraddloss', 'elecconsumpkeephot', 'cylinderstat', 'combinedcylinder', 'swhpumpsolarpowered',
                     'chargingbasisheatconsumed', 'deliveredlightingenergy', 'deliveredenergypumpsfans', 'deliveredenergymainwater',
                     'deliveredenergymainspace', 'primaryenergylighting', 'primaryenergypumpsfans', 'primaryenergymainwater',
                     'primaryenergymainspace', 'co2lighting', 'co2pumpsfans', 'co2mainwater', 'co2mainspace',
                     'groundfloorarea', 'groundfloorheight', 'firstfloorarea', 'firstfloorheight', 'secondfloorarea',
                     'secondfloorheight', 'thirdfloorarea', 'thirdfloorheight', 'thermalbridgingfactor', 'lowenergylightingpercent',
                     'deliveredenergysecondaryspace', 'livingareapercent', 'co2secondaryspace', 'primaryenergysecondaryspace',
                     'primaryenergysupplementarywater', 'roominroofarea', 'firstenerproddelivered', 'firstpartltotalcontribution',
                     'firstenerprodconvfactor', 'firstenerprodco2emissionfactor', 'firstenerconsumeddelivered', 'firstenerconsumedconvfactor',
                     'firstenerconsumedco2emissionfactor', 'secondenerproddelivered', 'secondenerprodconvfactor', 'secondenerprodco2emissionfactor',
                     'secondenerconsumeddelivered', 'secondenerconsumedconvfactor', 'secondenerconsumedco2emissionfactor',
                     'thirdenerproddelivered', 'thirdenerprodconvfactor', 'thirdenerprodco2emissionfactor', 'thirdenerconsumeddelivered',
                     'thirdenerconsumedconvfactor', 'thirdenerconsumedco2emissionfactor', 'totalprimaryenergyfact', 'totalco2emissions',
                     'firstwallarea', 'firstwalluvalue', 'firstwallissemiexposed', 'firstwallagebandid']

# Initialize the scaler
scaler = MinMaxScaler()

for col in tqdm(numerical_columns, desc="Scaling Numerical Columns"):
    if col in df.columns:  # Check if the column exists in the dataframe
        # Handle non-numeric values in the numerical columns
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric values to NaN
        if df[col].isnull().all():  # Check if the entire column is NaN
            print(f"Column '{col}' contains all NaN values. Skipping scaling for this column.")
            continue  # Skip this column
        
        # Fill NaNs with the column mean
        df[col] = df[col].fillna(df[col].mean())
        
        # Apply scaling
        df[col] = scaler.fit_transform(df[[col]])
    else:
        print(f"Column '{col}' not found in the dataset, skipping scaling.")

# Save the processed data to a new CSV file
cleaned_file_path = os.path.join(data_folder, "processed_data.csv")
df.to_csv(cleaned_file_path, index=False)

print("✅ Data processing completed and saved to 'processed_data.csv'.")
