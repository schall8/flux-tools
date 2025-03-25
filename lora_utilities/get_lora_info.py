# initial script
import os
import json
import argparse
from safetensors.torch import safe_open

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description="Inspect .safetensors LoRA files.")
parser.add_argument(
    "-d", "--directory",
    required=True,
    help="Directory containing .safetensors files"
)
args = parser.parse_args()

# Use the provided directory
lora_dir = args.directory

if not os.path.exists(lora_dir):
    print(f"❌ Directory not found: {lora_dir}")
    exit()

# Collect all .safetensors files
safetensor_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]

if not safetensor_files:
    print("No .safetensors files found in the directory.")
    exit()

# Display list of files
print("Available .safetensors files:")
for idx, filename in enumerate(safetensor_files):
    print(f"{idx + 1}. {filename}")

# User selects files by number
while True:
    selection = input(f"\nEnter file numbers to inspect (e.g. 1,3,5): ").strip()
    try:
        indices = [int(i) - 1 for i in selection.split(",") if i.strip().isdigit()]
        if all(0 <= i < len(safetensor_files) for i in indices):
            break
        else:
            print("Invalid numbers. Please try again.")
    except ValueError:
        print("Invalid input format. Please enter numbers separated by commas.")

selected_files = [safetensor_files[i] for i in indices]

# Ask how to display results
output_choice = input("\nOutput to screen or file? (screen/file): ").strip().lower()

results = {}

# Process each selected file
for filename in selected_files:
    file_path = os.path.join(lora_dir, filename)
    file_data = {}

    try:
        with safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
            keys = list(f.keys())

            file_data["metadata"] = metadata if metadata else {}
            file_data["tensor_keys"] = keys
    except Exception as e:
        file_data["error"] = str(e)

    results[filename] = file_data

# Show or save results
if output_choice == "file":
    file_format = input("Export as JSON or TXT? (json/txt): ").strip().lower()
    out_path = os.path.join(lora_dir, f"lora_metadata_export.{file_format}")

    try:
        if file_format == "json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        else:  # plain text
            with open(out_path, "w", encoding="utf-8") as f:
                for fname, data in results.items():
                    f.write(f"\n=== {fname} ===\n")
                    if "error" in data:
                        f.write(f"Error: {data['error']}\n")
                        continue
                    f.write("Metadata:\n")
                    for k, v in data["metadata"].items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\nTensor Keys:\n")
                    for key in data["tensor_keys"]:
                        f.write(f"  {key}\n")
        print(f"\n✅ Export complete: {out_path}")
    except Exception as e:
        print(f"❌ Failed to write file: {e}")

else:
    for fname, data in results.items():
        print(f"\n=== {fname} ===")
        if "error" in data:
            print(f"Error: {data['error']}")
            continue
        print("Metadata:")
        for k, v in data["metadata"].items():
            print(f"  {k}: {v}")
        print("\nTensor Keys:")
        for key in data["tensor_keys"]:
            print(f"  {key}")
