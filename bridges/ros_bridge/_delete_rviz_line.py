"""
Only for github CI using! The visualization tool is not available on github server, remove it.
"""
import os


def delete_line_from_file(file_name, line_number):
    # Check if the line number is valid
    if line_number < 1:
        return

    lines = []
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # Check if the line number is beyond the file's length
    if line_number > len(lines):
        return

    with open(file_name, 'w') as f:
        for idx, line in enumerate(lines):
            if idx + 1 != line_number:
                f.write(line)


# Usage
if __name__ == '__main__':
    delete_line_from_file(os.path.join(os.path.dirname(__file__), 'src', "metadrive_example_bridge", 'package.xml'), 10)
