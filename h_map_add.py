import pandas as pd

def h_map_add(path1, path2):
    """
    Adds corresponding elements of two CSV files and saves the result to a new CSV.

    Args:
        path1 (str): The file path of the first CSV.
        path2 (str): The file path of the second CSV.
    """
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
        output_path = 'h_map_add_output.csv'
        df_sum.to_csv(output_path, index=False)
        print(f"Success! The sum of the CSVs has been saved to '{output_path}'.")

    except FileNotFoundError:
        print("Error: One or both of the specified CSV files were not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage (assuming you have two CSV files named 'file1.csv' and 'file2.csv' in the same directory)
h_map_add('h_map_unif_60.csv', 'h_map_rand.csv')
