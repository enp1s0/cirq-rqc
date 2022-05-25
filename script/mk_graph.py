import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# Output
output_file_name = "figure.pdf"

# The list of `type` in the input csv file

# Figure config
plt.figure(figsize=(8, 3))
plt.xlabel("log(NP)")
plt.ylabel("Prob[log(NP)]")
#plt.yscale("log", base=10)
plt.grid()

# Load input data
df_t = pd.read_csv("data.csv", encoding="UTF-8")

print(df_t['amplitude'])
N = np.power(2., 49)
print('N =', N)
M = 10
print('M =', M)

accepted_samples = []

for a in df_t['amplitude']:
    a = complex(a)
    p = np.imag(a) * np.imag(a) + np.real(a) * np.real(a)

    Np = N * p

    prob_tmp = min(Np / M, 1)

    if random.random() < prob_tmp:
        accepted_samples += [Np]

print("Num sampled = ", len(accepted_samples))

min_Np = -6
max_Np = 2
resolution_Np = 10
prob_histogram = np.zeros((max_Np - min_Np + 1) * resolution_Np)

for s in accepted_samples:
    log_Np = np.log(s)

    x_index = int((log_Np - min_Np) * resolution_Np)

    if x_index > 0 and x_index < (max_Np - min_Np + 1) * resolution_Np:
        prob_histogram[x_index] += 1

plot_y = []
plot_x = []
for i in range((max_Np - min_Np + 1) * resolution_Np):
    plot_x += [(i / resolution_Np + min_Np)]
    v = prob_histogram[i] / float(len(accepted_samples)) * resolution_Np
    plot_y += [v if v != 0 else np.nan]

thepretical_y = [np.exp(x) * np.exp(-np.exp(x)) * np.exp(x) for x in plot_x]

# Plot
#plt.yscale("log", base=10)
plt.plot(plot_x, thepretical_y, label="theoretical", linewidth=1)
plt.plot(plot_x, plot_y, label="measured", linewidth=1, marker='+')

# Legend config
plt.legend(loc='best', ncol=1)

# Save to file
plt.savefig(output_file_name, bbox_inches="tight")
