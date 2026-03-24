import pandas as pd
import pyreadstat
import os
import numpy as np
from sklearn.impute import SimpleImputer
import os

DIR = '/Users/yashitamishra/Downloads/IMLNHANES/'
OUT = os.path.join(DIR, 'nhanes_diabetes_clean.csv')

files = {
    'DEMO'   : 'DEMO_J.xpt',    
    'BPX'    : 'BPX_J.xpt',     
    'BMX'    : 'BMX_J.xpt',     
    'GHB'    : 'GHB_J.xpt',     
    'BIOPRO' : 'BIOPRO_J.xpt',  
    'HDL'    : 'HDL_J.xpt',     
    'TCHOL'  : 'TCHOL_J.xpt',   
    'TRIGLY' : 'TRIGLY_J.xpt',  
}

dfs = {}
for name, filename in files.items():
    path = os.path.join(DIR, filename)
    df, _ = pyreadstat.read_xport(path)
    dfs[name] = df
    print(f"Loaded {name:8s}: {df.shape[0]:,} rows  {df.shape[1]} cols")

merged = dfs['DEMO'].copy()
for name in ['BPX', 'BMX', 'GHB', 'BIOPRO', 'HDL', 'TCHOL', 'TRIGLY']:
    merged = merged.merge(dfs[name], on='SEQN', how='left',suffixes=('', f'_{name}'))

print(f"\nMerged shape (all participants): {merged.shape}")

tier1 = [
    'RIDAGEYR',   
    'RIAGENDR',   
    'RIDRETH3',  
    'BMXBMI',     
    'BMXWAIST',   
    'BPXSY1',     
    'BPXDI1',     
    'BPXSY2',     
    'BPXDI2',     
]

tier2 = [
    'LBDHDL',     
    'LBXTC',      
    'LBXTR',      
    'LBXSCR',     
    'LBXSATSI',   
    'LBXSASSI',   
]

tier3 = [
    'LBXSUA',     
    'LBXSTP',     
    'LBXSAL',    
    'LBXSCA',    
    'LBXSPH',     
    'LBXSNASI',  
    'LBXSKSI',    
    'LBXSGB',     
    'LBXSBU',     
    'LBXSC3SI',   
]


label_col = ['LBXGH']   

all_wanted = ['SEQN'] + tier1 + tier2 + tier3 + label_col
present     = [c for c in all_wanted if c in merged.columns]
missing_col = [c for c in all_wanted if c not in merged.columns]

df = merged[present].copy()

before = len(df)
df = df.dropna(subset=['LBXGH'])
print(f"\nDropped {before - len(df):,} rows with missing HbA1c")
print(f"Remaining: {len(df):,} participants")

def hba1c_label(val):
    if val < 5.7:
        return 0   
    elif val < 6.5:
        return 1   
    else:
        return 2 

df['diabetes_label'] = df['LBXGH'].apply(hba1c_label)
df['diabetes_binary'] = (df['LBXGH'] >= 6.5).astype(int)  

print("\nClass distribution (3-class):")
counts = df['diabetes_label'].value_counts().sort_index()
labels_map = {0: 'Normal', 1: 'Prediabetic', 2: 'Diabetic'}
for k, v in counts.items():
    print(f"  {labels_map[k]:12s}: {v:,}  ({v/len(df)*100:.1f}%)")

if 'BPXSY1' in df.columns and 'BPXSY2' in df.columns:
    df['BPXSY_mean'] = df[['BPXSY1', 'BPXSY2']].mean(axis=1)
if 'BPXDI1' in df.columns and 'BPXDI2' in df.columns:
    df['BPXDI_mean'] = df[['BPXDI1', 'BPXDI2']].mean(axis=1)


if all(c in df.columns for c in ['LBXTC', 'LBDHDL', 'LBXTR']):
    mask = df['LBXTR'] < 400
    df['LDL_calc'] = np.nan
    df.loc[mask, 'LDL_calc'] = (
        df.loc[mask, 'LBXTC'] - df.loc[mask, 'LBDHDL'] - df.loc[mask, 'LBXTR'] / 5
    )
    print(f"\nLDL (Friedewald) calculated for {mask.sum():,} participants (TG < 400)")

feature_cols = [c for c in df.columns if c not in ['SEQN', 'LBXGH', 'diabetes_label', 'diabetes_binary']]

missing_pct = df[feature_cols].isnull().sum() / len(df) * 100
print("\nMissing % per feature (sorted):")
print(missing_pct[missing_pct > 0].sort_values(ascending=False).round(1).to_string())
imputer = SimpleImputer(strategy='median')
df[feature_cols] = imputer.fit_transform(df[feature_cols])

print(f"\nImputed {len(feature_cols)} feature columns with median strategy")

id_and_label = ['SEQN', 'LBXGH', 'diabetes_label', 'diabetes_binary']
df_full = df[id_and_label + [c for c in df.columns if c not in id_and_label]]
df_full.to_csv(OUT, index=False)
print(f"   nhanes_diabetes_clean.csv  — full dataset  {df_full.shape}")
print(f"\nTotal participants: {len(df):,}")