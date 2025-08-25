import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the CSV
df = pd.read_csv('your_data.csv', sep='\t')  # Use '\t' if it's tab-separated

# Step 2: Filter for one reservoir (e.g., 'Mornos')
reservoir_name = 'Mornos'
filtered = df[df['Reservoir'] == reservoir_name]

# Step 3: Extract the value series
series = filtered['Value'].values

# Step 4: Climacogram function
def compute_climacogram(series, max_scale):
    scales = np.arange(1, max_scale + 1)
    variances = []

    for scale in scales:
        chunks = [series[i:i+scale] for i in range(0, len(series), scale)]
        chunk_means = [np.mean(chunk) for chunk in chunks if len(chunk) == scale]
        var = np.var(chunk_means)
        variances.append(var)

    return scales, variances

# Step 5: Run and plot
scales, variances = compute_climacogram(series, max_scale=50)

plt.figure(figsize=(8, 5))
plt.loglog(scales, variances, marker='o')
plt.xlabel('Scale (window size)')
plt.ylabel('Variance of averaged values')
plt.title(f'Climacogram for {reservoir_name}')
plt.grid(True)
plt.show()