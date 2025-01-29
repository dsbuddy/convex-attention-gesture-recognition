import os
import json
import random
import numpy as np

# Data Processing
def process_data(filename, shuffle=True):
	print('Loading trials...')

	# Load data from JSON file
	dir = os.path.join('..', 'data')
	os.makedirs(dir, exist_ok=True)
	path = os.path.join(dir, filename)

	with open(path, 'r') as f:
		data = json.load(f)

	# Get unique labels
	seen_labels = set()
	ordered_labels = []
	for obj in data:
		label = obj['label']
		if label not in seen_labels:
			seen_labels.add(label)  # Mark label as seen
			ordered_labels.append(label)  # Add to ordered list
	labels = ordered_labels

	inputs_raw = [trial['data'] for trial in data]
	outputs_raw = [labels.index(trial['label']) for trial in data]

	# Shuffle data for better training
	combined = list(zip(inputs_raw, outputs_raw))
	if shuffle:
		random.shuffle(combined)
	inputs_raw, outputs_raw = zip(*combined) 

	# Convert to NumPy arrays for efficient processing
	inputs = np.array(inputs_raw)
	outputs = np.array(outputs_raw)
	
	mean = inputs.mean(axis=0)
	std = inputs.std(axis=0, ddof=1) + 1

	# Normalize inputs
	inputs = (inputs - mean) / (std)
	return inputs, outputs, mean, std, labels

# Header Generation
def format_header_data(data, output_header_name, array_name, rows, cols):
	with open(output_header_name, 'a') as outfile:  
		outfile.write(f'// Rows and columns for {array_name}\n')
		outfile.write(f'const size_t {array_name}_rows = {rows};\n')
		outfile.write(f'const size_t {array_name}_cols = {cols};\n')
		outfile.write(f'// Data for {array_name}\n')

		# Ensure data is a NumPy array of floats
		if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):  
			outfile.write(f'const float {array_name}[{rows}][{cols}] = {{\n')

			# Write the data row by row, formatting as floats
			for row in data:
				float_str = ', '.join(f'{val:.6f}f' for val in row)  # Adjust precision as needed
				outfile.write(f'\t{{{float_str}}},\n')

			outfile.write('};\n\n') 
		else:
			raise ValueError(f"Expected a NumPy array of floats, got: {data}")

def format_float_data(float_value, output_header_name, array_name):
	with open(output_header_name, 'a') as outfile:
		outfile.write(f"#define {array_name} {float_value}\n")

def dump_char_array_to_header(array_name, array_values, header_filename="model_data.h"):
	with open(header_filename, 'a') as outfile:

		# Array declaration
		outfile.write(f'extern const char* {array_name}[];\n')

		# Array size
		outfile.write(f'const int {array_name}_size = sizeof({array_name}) / sizeof({array_name}[0]);\n\n')

def dump_model_to_header(best_model, output_header_name='model_data.h', mean=None, std=None, labels=None):
    variables = [
        ("A", best_model['A'], best_model['A'].dtype.type),
        ("transformer_matrix", best_model['transformer'].transform_matrix, best_model['transformer'].transform_matrix.dtype.type),
        ("transformer_bias", best_model['transformer'].transform_bias, best_model['transformer'].transform_bias.dtype.type),
        ("transformer_n_components", best_model['transformer'].n_components, type(best_model['transformer'].n_components)),
		("mean", mean, mean.dtype.type),
		("stddev", std, std.dtype.type)
    ]

    dump_to_cpp_header(output_header_name, variables, labels)

def dump_to_cpp_header(filename, variables, labels=None):
    with open(filename, "w") as f:
        f.write("#include <stddef.h>\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#define LABELS char* outputLabels[] = {" + ",".join(f'"{direction}"' for direction in labels) + "}; // Array of output labels\n\n")
		
        for var_name, var_data, var_type in variables:
            # Array dimensions
            if isinstance(var_data, np.ndarray):
                # Convert float32 to float64
                if var_data.dtype == np.float32:
                    # Convert to float64
                    var_data = var_data.astype(np.float64)
                    var_type = np.float64
                rows, cols = var_data.shape
                f.write(f"// Rows and columns for {var_name}_data\n")
                f.write(f"int {var_name}_data_rows = {rows};\n")
                f.write(f"int {var_name}_data_cols = {cols};\n")

            # Data declaration
            cpp_type = {
                np.float64: "double",
                np.float32: "float",
                int: "int",
                float: "float"
            }[var_type]
            f.write(f"// Data for {var_name}_data\n")
            if isinstance(var_data, np.ndarray):
                f.write(f"{cpp_type} {var_name}_data[{rows}][{cols}] = {{\n")
                for row in var_data:
                    f.write("\t{")
                    f.write(", ".join(f"{val:.6f}f" if cpp_type == "float" else f"{val:.6f}" for val in row))
                    f.write("},\n")
                f.write("};\n\n")
            else:
                f.write(f"{cpp_type} {var_name}_data = {var_data:.6f};\n\n")

        f.write("#endif // MODEL_DATA_H\n")