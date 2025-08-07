import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('first_10graph.csv')

# Check the columns in the DataFrame
print(df.columns)

# Define the columns for the plot
data_columns = ['Simple Randomized or Randomized-1', 'Simple Greedy or Greedy-1', 
                'Semi Greedy - 1', 'Average Value', 'Best Value']

# Extract the graph names (G1, G2, ..., G10)
graphs = df['Problem'].values

# Extract the values for each algorithm
randomized = df['Simple Randomized or Randomized-1'].values
greedy = df['Simple Greedy or Greedy-1'].values
semi_greedy = df['Semi Greedy - 1'].values
local_search = df['Average Value'].values
grasp = df['Best Value'].values

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Set the width of the bars
bar_width = 0.15

# Define the positions of the bars
r1 = range(len(graphs))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Plot bars for each algorithm
ax.bar(r1, randomized, color='blue', width=bar_width, edgecolor='grey', label='Randomized')
ax.bar(r2, greedy, color='red', width=bar_width, edgecolor='grey', label='Greedy')
ax.bar(r3, semi_greedy, color='gray', width=bar_width, edgecolor='grey', label='Semi-Greedy')
ax.bar(r4, local_search, color='orange', width=bar_width, edgecolor='grey', label='Local Search')
ax.bar(r5, grasp, color='yellow', width=bar_width, edgecolor='grey', label='GRASP')

# Add labels
ax.set_xlabel('Graphs', fontweight='bold')
ax.set_ylabel('Max Cut Value', fontweight='bold')
ax.set_title('Max Cut (Graph 1-10)', fontweight='bold')
ax.set_xticks([r + bar_width * 2 for r in range(len(graphs))])
ax.set_xticklabels(graphs)

# Add a legend
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
