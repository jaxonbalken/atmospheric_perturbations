import random
import tkinter as tk
from tkinter import filedialog

# DAC settings
DAC_MIN = 0
DAC_MAX = 4096
DAC_CENTER = 2048
STEP_SIZE = 100        # Max change per step
DEFAULT_LENGTH = 3000  # Default number of DAC points

def bounded_random_walk(current_value, step_size=STEP_SIZE):
    step = random.randint(-step_size, step_size)
    new_value = current_value + step
    return max(DAC_MIN, min(DAC_MAX, new_value))

def choose_save_location():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.asksaveasfilename(
        title="Save DAC Output As",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    return file_path

def generate_dac_data(length=DEFAULT_LENGTH):
    x_value = DAC_CENTER
    y_value = DAC_CENTER

    file_path = choose_save_location()
    if not file_path:
        print("File save canceled.")
        return

    with open(file_path, "w") as file:
        file.write("x_DAC\ty_DAC\n")  # Header
        for _ in range(length):
            x_value = bounded_random_walk(x_value)
            y_value = bounded_random_walk(y_value)
            file.write(f"{x_value}\t{y_value}\n")

    print(f"Generated {length} DAC entries in '{file_path}'.")

if __name__ == "__main__":
    # You can adjust the number of points here
    generate_dac_data(length=3000)
