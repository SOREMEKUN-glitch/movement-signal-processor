import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

# 1. LOAD DATA 
df = pd.read_csv('Raw_Data.csv')
df.columns = ['time', 'accel_x', 'accel_y', 'accel_z', 'accel_abs']

time  = df['time'].values
accel = df['accel_abs'].values   # using absolute acceleration

# 2. CALCULATE SAMPLING RATE
fs = 1 / np.mean(np.diff(time))
print(f'Sampling rate: {fs:.1f} Hz')
print(f'Recording duration: {time[-1]:.1f} seconds')
print(f'Total samples: {len(accel)}')

# 3. LOW-PASS FILTER (remove noise) 
def lowpass_filter(data, cutoff=5, fs=50, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, data)

filtered = lowpass_filter(accel, cutoff=5, fs=fs)

# 4. EXTRACT FEATURES 
mean_intensity = np.mean(np.abs(filtered))
variability    = np.std(filtered)
jerk           = np.diff(filtered)

# Step detection
peaks, _  = find_peaks(filtered, height=np.mean(filtered), distance=int(fs * 0.4))
step_count = len(peaks)
cadence    = step_count / (time[-1] / 60)   # steps per minute

print(f'\n── Movement Features ──────────────────')
print(f'Mean intensity : {mean_intensity:.4f} m/s²')
print(f'Variability    : {variability:.4f} m/s²')
print(f'Steps detected : {step_count}')
print(f'Cadence        : {cadence:.1f} steps/min')

# 5. PLOT 
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Movement Signal Analysis — Oyindamola Soremekun', 
             fontsize=14, fontweight='bold')

# Plot 1: Raw vs Filtered
axes[0].plot(time, accel, alpha=0.4, color='gray', label='Raw signal')
axes[0].plot(time, filtered, color='steelblue', linewidth=1.5, label='Filtered signal')
axes[0].plot(time[peaks], filtered[peaks], 'rv', markersize=8, label=f'Steps detected ({step_count})')
axes[0].set_title('Absolute Acceleration: Raw vs Filtered with Step Detection')
axes[0].set_ylabel('Acceleration (m/s²)')
axes[0].legend()

# Plot 2: X, Y, Z axes
axes[1].plot(time, df['accel_x'], label='X axis', alpha=0.8)
axes[1].plot(time, df['accel_y'], label='Y axis', alpha=0.8)
axes[1].plot(time, df['accel_z'], label='Z axis', alpha=0.8)
axes[1].set_title('Acceleration by Axis (X, Y, Z)')
axes[1].set_ylabel('Acceleration (m/s²)')
axes[1].legend()

# Plot 3: Jerk (rate of movement change)
axes[2].plot(time[1:], jerk, color='darkorange', linewidth=1, label='Jerk')
axes[2].set_title('Jerk — Rate of Change in Movement (linked to movement quality)')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Jerk (m/s³)')
axes[2].legend()

plt.tight_layout()
plt.savefig('signal_plot.png', dpi=150)
plt.show()
print('\nPlot saved as signal_plot.png')
