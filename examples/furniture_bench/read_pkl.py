import pickle
import argparse
import numpy as np

def load_and_print_pkl(file_path):
    """
    Loads and prints the content of a .pkl file.

    Args:
        file_path (str): Path to the .pkl file.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Pretty-print the content
        print("Contents of the .pkl file:")
        if isinstance(data, dict):
            for key, value in data.items():
                # print(f"Key: {key}")
                # print(f"Value: {type(value)}")
                if key == "observations":
                    print("length:", len(value))
                    val = value[0]
                    if isinstance(val, dict):
                        for k, v in val.items():
                            print(f"    key: {k}")
                            print(f"    value: {type(v)}")
                            if isinstance(v, np.ndarray):
                                print(f"    value: {v.shape}")
                            if isinstance(v, dict):
                                for k2, v2 in v.items():
                                    print(f"        key: {k2}")
                                    print(f"        value: {type(v2)}")
                                    if isinstance(v2, np.ndarray):
                                        print(f"        value: {v2.shape}")
                                    else:
                                        print(f"        value: {v2}")
                    else:
                        print(val)
                elif key == "actions":
                    print("action length:", len(value[0]))
                    print(value[0])
        else:
            print(data)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
    except pickle.UnpicklingError:
        print(f"Error: Failed to load the pickle file. The file might be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Print the content of a .pkl file.")
    parser.add_argument("--path", type=str, required=True, help="Path to the .pkl file.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided path
    load_and_print_pkl(args.path)