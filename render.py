import os
import shutil
import zipfile
from flask import Flask, render_template, request, send_file
from keras2c.keras2c_main import *

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to clear all contents in the upload folder
def clear_upload_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def cleanup_generated_files():
    try:
        # Delete the .c, .h, and .test_suite.c files
        for filename in os.listdir():
            if filename.endswith(".c") or filename.endswith(".h"):
                os.remove(filename)
                print(f"Deleted {filename}")

       
    except Exception as e:
        print(f"Error cleaning up files for {filename}: {e}")

# Function to zip only specific files and the k2c folder
def create_zip(output_folder, zip_filename, function_name):
    zip_filepath = os.path.join(UPLOAD_FOLDER, zip_filename)

    # List of specific files to include (with dynamic renaming based on function_name)
    files_to_zip = [
        os.path.join(output_folder, f"{function_name}.c"),
        os.path.join(output_folder, f"{function_name}.h"),
        os.path.join(output_folder, f"{function_name}_test_suite.c"),
        os.path.join(output_folder, "k2c")  # Assuming k2c folder is generated
    ]
    
    # Create a Zip file and add only the specified files
    try:
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                if os.path.exists(file_path):
                    # Add the file or folder to the zip
                    if os.path.isdir(file_path):
                        # If it's a directory, add its contents
                        for root, dirs, files in os.walk(file_path):
                            for file in files:
                                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_folder))
                    else:
                        # If it's a file, add it directly
                        zipf.write(file_path, os.path.relpath(file_path, output_folder))
        return zip_filepath
    except Exception as e:
        print(f"Error creating zip file: {e}")
        return None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    # Clear all contents in the upload folder
    clear_upload_folder()
    cleanup_generated_files()

    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Get the function name from the form
    function_name = request.form.get('function_name')  # Get the function name from the form

    if file:
        # Save file to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Convert H5 file to C code using the user-provided function name
            k2c(filepath, function_name, malloc=False, num_tests=1, verbose=True)

            # Set the folder where the generated files are stored (assuming k2c creates them in the current directory)
            generated_folder = os.getcwd()  # Assuming k2c generates files in the current working directory

           
            # Create a zip file with only the specific files and k2c folder
            zip_filename = f"{function_name}_output.zip"
            zip_filepath = create_zip(generated_folder, zip_filename, function_name)

            # Check if the zip file was created successfully
            if zip_filepath is None:
                return "Error creating zip file."

            # Delete the original .h5 file after conversion
            os.remove(filepath)
            # Send the zip file to the user
            return send_file(zip_filepath, as_attachment=True)

        except Exception as e:
            return f"Error converting model: {e}"

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
