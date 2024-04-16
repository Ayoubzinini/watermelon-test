import numpy as np
import pandas as pd

# Define parameters
num_samples = 100
num_wavelengths = 1024
wavelength_start = 1350  # in nm
wavelength_end = 2550    # in nm

# Generate wavelength values
wavelengths = np.linspace(wavelength_start, wavelength_end, num_wavelengths)

# Generate spectral data for watermelon reflectance
spectral_data = np.random.rand(num_samples, num_wavelengths)

# Create DataFrame
df = pd.DataFrame(spectral_data, columns=wavelengths)

# Add column for Brix degree
# Generate random Brix degree for each sample
brix_degrees = np.random.uniform(low=7, high=12, size=num_samples)  # Assuming typical Brix degree range for watermelon

# Add Brix degree column to DataFrame
df['Brix_Degree'] = brix_degrees

# Optionally, you can round the wavelength values to make them more readable
df = df.round(2)

# Display the DataFrame
print(df)
