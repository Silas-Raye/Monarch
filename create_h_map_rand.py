import csv
import random

"""
Generates a CSV file named 'h_map.csv' with 26 rows and 30 columns.
The cells are populated with random integers between -10 and 0.
"""

# Define the number of rows and columns
num_rows = 26
num_cols = 30

# Define the file name
file_name = 'h_map_rand.csv'

# Open the file in write mode
with open(file_name, 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Iterate through the rows
    for _ in range(num_rows):
        # Generate a row of 30 random integers between -10 and 0
        row = [random.randint(-10, 0) for _ in range(num_cols)]
        # Write the row to the CSV file
        writer.writerow(row)

print(f"Successfully created '{file_name}' with {num_rows} rows and {num_cols} columns.")
