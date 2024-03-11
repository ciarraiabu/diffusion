import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('OTDR_over20.csv')
df2 = pd.read_csv('generated_data.csv')

df = df[~df['Class'].isin([4, 5])]

df2 = df2[['SNR', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'Max_Amplitude', 'Class']]

temp_df = pd.concat([df, df2])

temp_df.to_csv('synthetic_over20.csv', index=False)

grouped = df.groupby('Class')

class_statistics = {}
for name, group in grouped:
    statistics = {
        'Class': name,
        #'SNR_Min': group['SNR'].min(),
        #'SNR_Max': group['SNR'].max(),
        #'SNR_Mean': group['SNR'].mean(),
        #'SNR_Median': group['SNR'].median(),
        #'SNR_StdDev': group['SNR'].std(),
        'Amp_Min': group['Max_Amplitude'].min(),
        'Amp_Max': group['Max_Amplitude'].max(),
        'Amp_Mean': group['Max_Amplitude'].mean(),
        'Amp_Median': group['Max_Amplitude'].median(),
        'Amp_StdDev': group['Max_Amplitude'].std(),
    }
    class_statistics[name] = statistics

# Convert the statistics to a DataFrame
statistics_df = pd.DataFrame(list(class_statistics.values()))

#%%

df1 = pd.read_csv('OTDR_diffusion.csv')

# Criteria for selection
selected_class = 4  # Example class
selected_snr = 20  # Example SNR level
selected_max_amp = 10  # Example max_amp

# Filter signals based on criteria for each dataframe
signal_df1 = df1[(df1['Class'] == selected_class) &
                 (df1['SNR'] == selected_snr) &
                 (df1['Max_Amplitude'] == selected_max_amp)].iloc[0]

signal_df2 = df2[(df2['Class'] == selected_class) &
                 (df2['SNR'] == selected_snr) &
                 (df2['Max_Amplitude'] == selected_max_amp)].iloc[0]

# Assuming the signals are stored in columns named P1 to P30
signal_columns = [f'P{i}' for i in range(1, 31)]
signal1 = signal_df1[signal_columns].values
signal2 = signal_df2[signal_columns].values

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # First subplot
plt.plot(signal1, label=f'Class {selected_class}, SNR {selected_snr}, Max Amp {selected_max_amp}')
plt.title('Real OTDR Trace')
plt.legend()

plt.subplot(1, 2, 2)  # Second subplot
plt.plot(signal2, label=f'Class {selected_class}, SNR {selected_snr}, Max Amp {selected_max_amp}')
plt.title('Generated OTDR Trace')
plt.legend()

plt.tight_layout()
plt.show()