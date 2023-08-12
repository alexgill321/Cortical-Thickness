from utils import generate_data_thickness_only_analysis
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#%%
filename = os.path.join(os.getcwd(), 'data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train, test, columns = generate_data_thickness_only_analysis(filename, normalize=False)
#%%
(train_x, train_y) = train
(test_x, test_y) = test

#%%
# Select the first 10 features from train_x
selected_features = train_x[:, :10]

fig, axs = plt.subplots(5, 2, figsize=(10, 15))
axs = axs.flatten()

for i in range(10):
    sns.histplot(selected_features[:, i], kde=False, ax=axs[i])
    axs[i].set_title(f"Feature {i+1}")

plt.tight_layout()
plt.show()

#%% Calculate mean and std for each feature

mean = train_x.mean(axis=0)
std = train_x.std(axis=0)

total_mean = mean.mean()
total_std = std.mean()
print(f"Total mean: {total_mean}")
print(f"Total std: {total_std}")
plt.hist(mean)
print("Normalized Features")



#%%
train, test, columns = generate_data_thickness_only_analysis(filename, normalize=False)

(train_x_no_norm, train_y) = train
(test_x_no_norm, test_y) = test

#%%
# Select the first 10 features from train_x
selected_features = train_x_no_norm[:, :10]

fig, axs = plt.subplots(5, 2, figsize=(10, 15))
axs = axs.flatten()

for i in range(10):
    sns.histplot(selected_features[:, i], kde=False, ax=axs[i])
    axs[i].set_title(f"Feature {i+1}")

plt.tight_layout()
plt.show()

#%% Calculate mean and std for each feature

mean = train_x_no_norm.mean(axis=0)
std = train_x_no_norm.std(axis=0)
plt.hist(mean)
# Show the mean and std for each feature
print("Un-normalized Features")
for i in range(100):
    print(f"{columns[i]}: mean={mean[i]}, std={std[i]}")


