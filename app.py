import base64
from flask import Flask, jsonify, request, render_template
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from langdetect import detect
import PyPDF2
from datetime import datetime
import random
import string
import smtplib
from email.mime.text import MIMEText
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from scipy.spatial.distance import cosine
from io import BytesIO
import pickle
from keras.layers import Flatten
from PIL import Image

app = Flask(__name__, static_folder='static')

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Define your email credentials
email_address = 'banusaibuddhula@gmail.com'
email_password = 'wwwy spus ghxq bvnt'
# Load the trained machine learning model
model = joblib.load(r"D:\Project Datasets\Saved_models\Saved_models\plagiarism_model.pkl")

# Load the precomputed reference vectors
precomputed_reference_vectors = joblib.load(r"D:\Project Datasets\Saved_models\precomputed_reference_vectors.pkl")

# Load the pretrained TF-IDF vectorizer
pretrained_tfidf_vectorizer = joblib.load(r"D:\Project Datasets\Saved_models\tfidf_vectorizer.pkl")

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf = PyPDF2.PdfReader(pdf_file)
    for page in range(len(pdf.pages)):
        text += pdf.pages[page].extract_text()
    return text



def calculate_cosine_similarity(doc1, doc2, tfidf_vectorizer):
    tfidf_matrix = tfidf_vectorizer.transform([doc1, doc2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim

def preprocess_text(text, language):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]

    if language == 'en':
        language_stopwords = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in language_stopwords]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def highlight_matching_words_in_text(user_text, reference_text):
    user_tokens = user_text.split()
    reference_tokens = reference_text.split()
    common_words = set(user_tokens) & set(reference_tokens)

    highlighted_text = []
    for word in user_tokens:
        if word in common_words:
            highlighted_text.append(f"<span style='color:red'>{word}</span>")
        else:
            highlighted_text.append(word)

    return ' '.join(highlighted_text)

def send_email(recipient, subject, message):
    msg = MIMEText(message)
    msg['From'] = email_address
    msg['To'] = recipient
    msg['Subject'] = subject

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_address, email_password)
        server.sendmail(email_address, recipient, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Email not sent:", e)
        return False

@app.route('/')
def index():
    return render_template('index.html', plagiarism_message=None)

@app.route('/check_image_plagiarism', methods=['POST'])
def check_image_similarity():
    try:
        # Load VGG16 model
        print("Loading VGG16 model...")
        base_model = VGG16(weights='imagenet', include_top=False)
        model = Model(inputs=base_model.input, outputs=Flatten()(base_model.layers[-1].output))

        # Load the user-uploaded image
        uploaded_image = request.files['image']  # Assuming the user uploads an image

        if not uploaded_image:
            return "No image provided."

        image_bytes = uploaded_image.read()

        user_x = image.img_to_array(image.load_img(BytesIO(image_bytes), target_size=(224, 224)))
        user_x = np.expand_dims(user_x, axis=0)
        user_x = preprocess_input(user_x)

        # Extract features from the user-uploaded image
        print("Extracting features from the user-uploaded image...")
        user_features = model.predict(user_x)

        # Specify the folder containing multiple images
        image_folder = r"D:\Project Datasets\din"

        # Load the saved feature dictionary
        saved_features_path = r"D:\Project Datasets\Saved_models/image_features_dict.pkl"  # Replace with the path to your saved feature dictionary
        print(f"Loading saved features from {saved_features_path}...")
        with open(saved_features_path, 'rb') as file:
            saved_features_dict = pickle.load(file)

        # Function to calculate cosine similarity between two feature vectors
        def calculate_cosine_similarity(feature1, feature2):
            return cosine_similarity(feature1, feature2)[0][0]

        # Calculate cosine similarity with saved features
        print("Calculating cosine similarity with saved features...")
        similarities = {}
        for image_file, saved_features in saved_features_dict.items():
            similarity = calculate_cosine_similarity(user_features, saved_features)
            similarities[image_file] = similarity

        # Find the file name with the highest cosine similarity
        most_similar_image = max(similarities, key=similarities.get)
        high_similarity = similarities[most_similar_image]
        highest_similarity=high_similarity * 100

        print(f"The most similar image is {most_similar_image} with a cosine similarity of {highest_similarity:.4f}")

        # Build the paths for user-uploaded image and the most similar image
        image_folder_path = r"D:\Project Datasets\din"
        most_similar_image_path = os.path.join(image_folder_path, most_similar_image)

        # Load the most similar image
        most_similar_img = Image.open(most_similar_image_path)

    # Convert the most similar image to base64 format
        most_similar_img_data = image_to_base64(most_similar_img)

       

        return render_template('image_plagiarism.html', highest_similarity=highest_similarity, 
               most_similar_image=most_similar_img_data,
               uploaded_image=uploaded_image,  # Pass the uploaded image
               most_similar_img_data=most_similar_img_data)



    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return error_message

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    try:
        user_text_manual = request.form.get('user_text_manual')

        if user_text_manual:
            user_text = user_text_manual
        else:
            uploaded_file = request.files['file_upload']
            if uploaded_file:
                content_type = uploaded_file.content_type

                if content_type.startswith('text'):
                    user_text = uploaded_file.read().decode('utf-8')
                elif content_type == 'application/pdf':
                    user_text = extract_text_from_pdf(uploaded_file)
                else:
                    return "The uploaded file is not a supported file type."

            else:
                return "No input text or file provided."

        detected_language = detect_language(user_text)
        user_text_preprocessed = preprocess_text(user_text, detected_language)

        max_similarity = -1
        most_similar_file = None
        most_similar_content = ""
        plagiarism_detected = False

        for filename, reference_data in precomputed_reference_vectors.items():
            reference_vector = reference_data["vector"]
            reference_text = reference_data["content"]

            similarity = calculate_cosine_similarity(user_text_preprocessed, reference_text, pretrained_tfidf_vectorizer)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_file = filename
                most_similar_content = reference_text

            if similarity > 0.6:  # Set desired similarity threshold
                plagiarism_detected = True
                break

        highlighted_user_text = highlight_matching_words_in_text(user_text, most_similar_content)

        if plagiarism_detected:
            plagiarism_message = f"High Plagiarism Detected ({max_similarity * 100:.2f}%)"
        elif max_similarity > 0:
            plagiarism_message = f"Less Plagiarism Detected ({max_similarity * 100:.2f}%)"
        else:
            plagiarism_message = "No Plagiarism Detected"

        # Check if the "update_vectors" checkbox is selected
        update_vectors_checkbox = request.form.get("update_vectors")
        if update_vectors_checkbox == "yes":
            if max_similarity < 0.5:
                # Generate a unique filename and update reference vectors
                unique_filename = f"user_input_{datetime.now().strftime('%Y%m%d%H%M%S')}{''.join(random.choice(string.ascii_letters) for _ in range(4))}.txt"
                precomputed_reference_vectors[unique_filename] = {
                    'vector': pretrained_tfidf_vectorizer.transform([user_text_preprocessed]).toarray()[0],
                    'content': user_text
                }
                joblib.dump(precomputed_reference_vectors,r"D:\Project Datasets\Saved_models\precomputed_reference_vectors.pkl")
        most_similar_filename = os.path.basename(most_similar_file)
        # Check if the "email" field is provided
        recipient_email = request.form.get("email")
        if recipient_email:
            # Send an email with the plagiarism report
            subject = "Plagiarism Report"
            message = f"Hello,\n\nThis is your plagiarism report regarding the file {uploaded_file.filename}.\n"
            message += f"Similarity Percentage: {max_similarity * 100:.2f}%\n"
            message += f"The most similar file: {most_similar_filename}\n"
            if request.form.get('update_vectors') == 'yes':
                message += "Thank you for giving access to your document content."
            else:
                message += "Thank you."

            send_email(recipient_email, subject, message)

        return render_template('text_plagiarism.html', plagiarism_message=plagiarism_message, highlighted_text=highlighted_user_text)

    except Exception as e:
        return str(e)

@app.route('/text_plagiarism.html')
def text_plagiarism():
    return render_template('text_plagiarism.html')

@app.route('/image_plagiarism.html')
def image_plagiarism():
    return render_template('image_plagiarism.html')

@app.route('/signup.html')
def login():
    return render_template('signup.html')



if __name__ == '__main__':
    app.run(debug=True)
