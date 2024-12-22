import os
from flask import Flask, request, jsonify, render_template 
from dotenv import load_dotenv
from flask_cors import CORS
import joblib
import requests
import whisper
import pandas as pd
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin 
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

class FitTransformPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def fit(self, X, y):
        X_transformed = self.preprocessor.fit_transform(X, y)
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)

    def fit_transform(self, X, y):
        return self.preprocessor.fit_transform(X, y)
    
class ConvertToDatetime(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy() 
        
        if 'Call Time' in X.columns:
            if not pd.api.types.is_datetime64_any_dtype(X['Call Time']):
                X['Call Time'] = pd.to_datetime(X['Call Time'],format='%H:%M:%S', errors='coerce')
        else:
            raise ValueError("The input DataFrame must contain a 'Call Time' column.")
        
        return X

class StemmingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.port_stem = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, (list, pd.Series)):
            raise ValueError("Input must be a pandas Series or a list of strings.")

        return X.apply(self._stem_content)
    
    def _stem_content(self, content):
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [
            self.port_stem.stem(word) 
            for word in stemmed_content 
            if word not in self.stop_words
        ]
        return ' '.join(stemmed_content)
    
class SqueezeTextColumn(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                X = X.squeeze(axis=1)
            else:
                raise ValueError("Input must be a single-column DataFrame for squeezing.")
        elif not isinstance(X, pd.Series):
            raise TypeError("Input must be a pandas Series or single-column DataFrame.")
        
        return X

class TimeOfDayTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if 'Hour' not in X.columns:
            raise ValueError("The input DataFrame must contain an 'Hour' column.")

        X['TimeOfDay'] = pd.cut(X['Hour'], bins=[0, 7, 9, 12, 15, 17, 20, 23], labels=['0-7', '7-9', '9-12', '12-15','15-17','17-20','20-23'], right=False)

        return X
    
class AddHourDropTime(BaseEstimator, TransformerMixin):
    def fit(self,X,y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['Hour'] = X['Call Time'].dt.hour.astype(int)

        X.drop(columns='Call Time', inplace=True,  errors='ignore')

        return X
    

class AddAreaCodeDropNumber(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        X['Area Code'] = X['Caller Number'].str[3:6]

        def clean_phone_number(phone_number):
            cleaned_number = re.sub(r'[^0-9]', '', phone_number)
            return cleaned_number[-7:]

        X['PhNoLastDig'] = X['Caller Number'].apply(clean_phone_number)

        X['Area Code'] = X['Area Code'].astype(int)
        X['PhNoLastDig'] = X['PhNoLastDig'].astype(int)
        
        X.drop(columns=['Caller Number'], inplace=True, errors='ignore')
        
        
        return X
    
load_dotenv()
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

whisper_model = whisper.load_model("base")
try:
    spam_model = joblib.load('./model_pipeline.joblib')
    # print(spam_model)
except FileNotFoundError:
    print("Spam detection model not found. Please ensure the model is trained and saved.")
    spam_model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/transcribe-audio', methods=['POST'])
def transcribe_audio():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    if spam_model is None:
        return jsonify({"error": "Spam detection model not loaded"}), 500

    try:
        caller_number = request.form.get('phoneNumber', '')
        call_duration = request.form.get('callDuration', '0')
        call_frequency_day = request.form.get('frequencyPerDay', '1')
        call_frequency_week = request.form.get('frequencyPerWeek', '7')

        transcription = None
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
            audio_path = os.path.join('uploads', audio_filename)
            audio_file.save(audio_path)
            
            try:
                result = whisper_model.transcribe(audio_path)
                transcription = result['text']
                os.remove(audio_path)
            except Exception as e:
                return jsonify({"error": f"Transcription error: {str(e)}"}), 500
        
        elif 'conversationText' in request.form:
            transcription = request.form['conversationText']

        if not transcription:
            return jsonify({"error": "No transcription available"}), 400

        prediction_data = pd.DataFrame({
            'Caller Number': [caller_number],
            'Call Time': [pd.to_datetime(datetime.now().strftime('%H:%M:%S'), format='%H:%M:%S')],
            'Call Duration(in s)': [float(call_duration)],
            'Call Frequency Per Day': [int(call_frequency_day)],
            'Call Frequency Per Week': [int(call_frequency_week)],
            'Conversation': [transcription]
        })

        print(prediction_data)
        prediction = spam_model.predict(prediction_data)[0]
        print(prediction)

        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"Error in transcribe-audio: {str(e)}")
        return jsonify({
            "error": f"Prediction process error: {str(e)}"
        }), 500


@app.route('/get-location', methods=['POST'])
def get_location():
    data = request.get_json()
    phone_number = data.get('phone_number', '')

    if not phone_number:
        return jsonify({"error": "Phone number not provided"}), 400

    numeric_phone = ''.join(filter(str.isdigit, phone_number))

    if len(numeric_phone) < 10:
        return jsonify({"error": "Invalid phone number format"}), 400

    if numeric_phone.startswith('1'):
        area_code = numeric_phone[1:4]
    else:
        area_code = numeric_phone[:3]

    city_names = {
        "305": "Miami, FL",
        "212": "New York, NY",
        "917": "New York, NY",
        "213": "Los Angeles, CA",
        "312": "Chicago, IL", 
        "415": "San Francisco, CA",
        "702": "Las Vegas, NV", 
        "617": "Boston, MA", 
        "404": "Atlanta, GA", 
        "512": "Austin, TX", 
        "713": "Houston, TX", 
        "808": "Honolulu, HI",
        "206": "Seattle, WA", 
        "602": "Phoenix, AZ", 
        "614": "Columbus, OH", 
        "919": "Raleigh, NC",
        "303": "Denver, CO", 
        "907": "Anchorage, AK",
        "406": "Billings, MT", 
        "208": "Boise, ID", 
        "605": "Sioux Falls, SD"
    }

    current_location = city_names.get(area_code)
    if not current_location:
        return jsonify({"error": f"Area code {area_code} not found"}), 404

    try:
        response = requests.get(
            f"https://api.opencagedata.com/geocode/v1/json?q={current_location}&key={OPENCAGE_API_KEY}"
        )
        response.raise_for_status()
        data = response.json()

        if data["results"]:
            lat = data["results"][0]["geometry"]["lat"]
            lng = data["results"][0]["geometry"]["lng"]
            return jsonify({"city": current_location, "latitude": lat, "longitude": lng})
        else:
            return jsonify({"error": "Location not found"}), 404

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True, port=3000)