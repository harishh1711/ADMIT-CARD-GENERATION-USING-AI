import glob
import cv2
import os
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Add 'os' module to the template context
app.jinja_env.globals['os'] = os

def detect_and_save_faces(image_path, output_dir, extension_factor=1.5):
    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale (face detection works on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each detected face as a separate image with extended boundary
    for i, (x, y, w, h) in enumerate(faces):
        # Extend the boundary by the extension_factor
        extended_x = max(0, int(x - (extension_factor - 1) * w / 2))
        extended_y = max(0, int(y - (extension_factor - 1) * h / 2))
        extended_w = min(image.shape[1] - extended_x, int(w * extension_factor))
        extended_h = min(image.shape[0] - extended_y, int(h * extension_factor))

        face_image = image[extended_y:extended_y + extended_h, extended_x:extended_x + extended_w]
        output_path = os.path.join(output_dir, f"face_{i+1}.jpg")
        cv2.imwrite(output_path, face_image)

    return [os.path.basename(face_path) for face_path in glob.glob(os.path.join(output_dir, "*.jpg"))]

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'file' in request.files:
        # Get the uploaded image file from the form
        image_file = request.files['file']

        # Save the uploaded image to a temporary directory
        temp_image_path = "temp_image.jpg"
        image_file.save(temp_image_path)

        # Define the output directory for saving the detected faces
        output_directory = "C:/Users/Swetha/Desktop/bithacks/static/output_faces"
        
        # Perform face detection and save the detected faces
        face_files = detect_and_save_faces(temp_image_path, output_directory)

        # Remove the temporary image file
        os.remove(temp_image_path)

        # Redirect to the result page after face detection
        return redirect(url_for('result', face_files=face_files))

    # Pass the 'os' module to the template context explicitly
    return render_template('upload_form.html', os=os)

@app.route('/result')
def result():
    # Get the list of face files in the output directory
    output_directory = "C:/Users/Swetha/Desktop/bithacks/static/output_faces"
    face_files = os.listdir(output_directory)

    # Render the result template, which will display the detected faces
    return render_template('result.html', face_files=face_files)


if __name__ == "__main__":
    app.run(debug=True)
