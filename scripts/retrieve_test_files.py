import os

# Define paths
directory_path = "../ABAW_7th/cropped_aligned_test"
output_file = "test_set_images.txt"

crawled_files = []
for root, _, files in os.walk(directory_path):
    for file_name in files:
        relative_path = os.path.relpath(os.path.join(root, file_name), directory_path)
        crawled_files.append(relative_path.replace("\\", "/"))

with open(output_file, "w") as output:
    output.write("\n".join(crawled_files))

print(f"Remaining files saved to {output_file}")
