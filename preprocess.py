import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('OTDR_minus3and5.csv')

#%%

df = df.reindex(['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','Class','SNR','Max_Amplitude'], axis=1)
df.SNR = df.SNR.round()
df['Class'] = df['Class'].astype('int')

#%%

# First, replace the numerical class labels with the fault names
class_mapping = {0: 'Normal -0', 1: 'Fiber Tapping -1', 2: 'Bad Splice -2', 3: 'Bending Event -3', 4: 'Dirty Connector -4', 5: 'Fiber Cut -5', 6: 'PC Connector -6', 7: 'Reflector -7'}
df['fault_name'] = df['Class'].map(class_mapping)

# Now, when you plot, use 'fault_name' for the x-axis
plt.figure(figsize=(12, 5))

# Subplot for SNR
plt.subplot(2, 1, 1)
sns.boxplot(x='fault_name', y='SNR', data=df)
plt.title('Distribution of SNR by Fault Class')

# Subplot for Max Amplitude
plt.subplot(2, 1, 2)
sns.boxplot(x='fault_name', y='Max_Amplitude', data=df)
plt.title('Distribution of Max Amplitude by Fault Class')

plt.tight_layout()
plt.show()

#%%

# Filter the DataFrame for rows where 'snr' > 10
filtered_df = df[df['SNR'] < 5]

# Count the number of traces in each class
class_counts = filtered_df['Class'].value_counts()

print(class_counts)
