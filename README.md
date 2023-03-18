# Facial Emotion Recognition
Facial Emotion Recognition is a Python-based project that utilizes Deep Learning models to recognize human facial expressions.
<br><br>

**The project includes the following features:**

* An interface that captures the user's webcam feed and applies the trained model to recognize the facial expression of the user in real-time.
* A trained Deep Learning model for facial expression recognition that has been trained on the FER2013 dataset using Keras with a Tensorflow backend.
* A Python script to train the model on the FER2013 dataset, and a Jupyter notebook with the model's training process.
* A requirements file that includes all the Python packages needed to run the project.
## Installation
1. Clone the repository:
```bash
git clone https://github.com/OmarEhab007/Facial-emotion-recognition.git
```
2. Change the directory to the project's root directory:
```bash
cd Facial-emotion-recognition
```
3. Create a virtual environment:
```bash
python3 -m venv env
```
4. Activate the virtual environment:
```bash
source env/bin/activate
```
5. Install the required Python packages:
```bash
pip install -r requirements.txt
```
6. Run the application:
```bash
python3 facial_expression_recognition.py
```
## Usage
* The application will prompt the user to allow access to the webcam.
* The user can then look into the camera, and the model will recognize their facial expression.
* The recognized facial expression will be displayed on the screen.
## Training
* The train.py script can be used to train the model on the FER2013 dataset.
* The Jupyter notebook facial_expression_recognition.ipynb shows the model's training process step-by-step and includes visualizations of the data and the model's performance.
## Credits
* The FER2013 dataset was created by Pierre-Luc Carrier and Aaron Courville, and can be found on **Kaggle**.
* The trained model architecture is based on the **VGG16** architecture.
* The code for capturing and processing the webcam feed is based on the OpenCV library.
