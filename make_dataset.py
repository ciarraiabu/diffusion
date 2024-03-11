import pandas as pd
import os
from sklearn.model_selection import train_test_split

current_working_directory = os.getcwd()
print("Current Working Directory:", current_working_directory)

# Load your dataset
df = pd.read_csv('OTDR_traces_processed.csv')



df['SNR'] = df['SNR'].astype('int')
#%%
df = df[~df['Class'].isin([3, 5])]

#%%

#df.to_csv('OTDR_minus3and5.csv', index=False)

#%%
df = df.drop(columns=[ 'Position', 'Reflectance', 'loss', 'max_value'])

#%%
df = df[df['SNR'] > 20]
#df = df[~((df['SNR'] > 5) & df['Class'].isin([1,4]))]
#%%
print(df['Class'].value_counts())
codes_column1, uniques_column1 = pd.factorize(df['Class'])
codes_column2, uniques_column2 = pd.factorize(df['Max_Amplitude'])

df['Class'] = codes_column1
df['Max_Amplitude'] = codes_column2
print(df['Class'].value_counts())
X = df.drop(columns='Class')
y = df['Class']
# Stratified split based on the combined key
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)

# Concatenate X_temp with y_temp into another DataFrame
temp_df = pd.concat([X_temp, y_temp], axis=1)

train_df.to_csv('OTDR_over20.csv', index=False)
temp_df.to_csv('OTDR_over20_val.csv', index=False)
#df.to_csv('OTDR_diffusion.csv', index=False)
