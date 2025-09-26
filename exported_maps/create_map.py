import csv
import random
import pandas as pd
import os

def create_map(num_rows, num_cols, file_name, out_dir="exported_maps", default_value=0, randomize=False, random_min=None, random_max=None, seed=None):
    """
    Creates a CSV file with a specified number of rows and columns,
    populating each cell with a default value.
    
    Args:
        num_rows (int): Number of rows in the CSV.
        num_cols (int): Number of columns in the CSV.
        file_name (str): Name of the output CSV file.
        default_value (int, optional): Value to populate each cell. Defaults to 0.
        randomize (bool, optional): If True, populate cells with random integers instead of default_value. Defaults to False.
        random_min (int, optional): Minimum value for random integers (inclusive). Required if randomize is True.
        random_max (int, optional): Maximum value for random integers (inclusive). Required if randomize is True.
        seed (int, optional): Seed for the random number generator to ensure reproducibility.
    """

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Full path for the output file
    file_path = os.path.join(out_dir, file_name)

    # Set the seed for the random number generator if provided
    if seed is not None:
        random.seed(seed)
        
    # Open the file in write mode
    with open(file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Iterate through the rows
        for _ in range(num_rows):
            # Generate a row with the default value
            if not randomize:
                row = [default_value for _ in range(num_cols)]
            else:
                row = [random.randint(random_min, random_max) for _ in range(num_cols)]
            # Write the row to the CSV file
            writer.writerow(row)

    print(f"Successfully created '{file_name}' with {num_rows} rows and {num_cols} columns.")

def map_add(io_dir = "exported_maps", path1_name = None, path2_name = None, output_name = "map_sum.csv"):
    """
    Adds corresponding elements of two CSV files and saves the result to a new CSV.

    Args:
        path1 (str): The file path of the first CSV.
        path2 (str): The file path of the second CSV.
        output_name (str): The name of the output CSV file. Defaults to "map_sum.csv".
    """

    # Ensure the directory exists
    os.makedirs(io_dir, exist_ok=True)
    path1 = os.path.join(io_dir, path1_name)
    path2 = os.path.join(io_dir, path2_name)
    output_path = os.path.join(io_dir, output_name)

    try:
        # Read the CSV files into pandas DataFrames
        df1 = pd.read_csv(path1, header=None)
        df2 = pd.read_csv(path2, header=None)
        
        # Check if the DataFrames have the same shape (rows and columns)
        if df1.shape != df2.shape:
            print("Error: The CSV files do not have the same number of rows and columns.")
            return
        
        # Add the corresponding elements of the two DataFrames
        df_sum = df1.add(df2)
        
        # Save the resulting DataFrame to a new CSV file
        df_sum.to_csv(output_path, index=False, header=None)
        print(f"Success! The sum of the CSVs has been saved to '{output_path}'.")

    except FileNotFoundError:
        print("Error: One or both of the specified CSV files were not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# create_map(12, 12, 'h_map_rand.csv', randomize=True, random_min=0, random_max=20, seed=10)
# create_map(12, 12, 'h_map_unif_30.csv', default_value=30)
# map_add(path1_name = 'h_map_rand.csv', path2_name = 'h_map_unif_30.csv', output_name = 'h_map_sum.csv')
# create_map(12, 12, 'viz_map_unif.csv', default_value=1)
create_map(12, 12, 'viz_map_cust.csv', default_value=0)
