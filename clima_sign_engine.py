import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from sklearn.preprocessing import LabelEncoder
import requests
from geopy.distance import geodesic
import time
import os
from PIL import Image
from stable_diffusion_pytorch import pipeline, model_loader
import subprocess
import json
import sys
import io
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("WEATHER_API_KEY")

if not API_KEY:
    raise ValueError("Missing WEATHER_API_KEY. Make sure it's set in the .env file.") #helper if not detected


# Force stdout and stderr to UTF-8 (for writing to logs reliably)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

UK_CITIES = [
    # Expanded UK cities for better weather variety
    {"name": "Aberdeen", "lat": 57.1497, "lon": -2.0943, "country": "UK"},
    {"name": "Inverness", "lat": 57.4778, "lon": -4.2247, "country": "UK"},
    {"name": "Plymouth", "lat": 50.3755, "lon": -4.1427, "country": "UK"},
    {"name": "Exeter", "lat": 50.7184, "lon": -3.5339, "country": "UK"},
    {"name": "Norwich", "lat": 52.6309, "lon": 1.2974, "country": "UK"},
    {"name": "Hull", "lat": 53.7676, "lon": -0.3274, "country": "UK"},
    {"name": "Dundee", "lat": 56.4620, "lon": -2.9707, "country": "UK"},
    {"name": "Luton", "lat": 51.8787, "lon": -0.4200, "country": "UK"},
    {"name": "Stoke-on-Trent", "lat": 53.0027, "lon": -2.1794, "country": "UK"},
    {"name": "Swansea", "lat": 51.6214, "lon": -3.9436, "country": "UK"},
    {"name": "London", "lat": 51.5074, "lon": -0.1276, "country": "UK"},
    {"name": "Manchester", "lat": 53.4808, "lon": -2.2426, "country": "UK"},
    {"name": "Birmingham", "lat": 52.4862, "lon": -1.8904, "country": "UK"},
    {"name": "Liverpool", "lat": 53.4084, "lon": -2.9916, "country": "UK"},
    {"name": "Leeds", "lat": 53.8008, "lon": -1.5491, "country": "UK"},
    {"name": "Glasgow", "lat": 55.8642, "lon": -4.2518, "country": "UK"},
    {"name": "Edinburgh", "lat": 55.9533, "lon": -3.1883, "country": "UK"},
    {"name": "Bristol", "lat": 51.4545, "lon": -2.5879, "country": "UK"},
    {"name": "Cardiff", "lat": 51.4816, "lon": -3.1791, "country": "UK"},
    {"name": "Belfast", "lat": 54.5973, "lon": -5.9301, "country": "UK"},
    {"name": "Sheffield", "lat": 53.3811, "lon": -1.4701, "country": "UK"},
    {"name": "Newcastle", "lat": 54.9784, "lon": -1.6174, "country": "UK"},
    {"name": "Southampton", "lat": 50.9097, "lon": -1.4044, "country": "UK"},
    {"name": "Nottingham", "lat": 52.9548, "lon": -1.1581, "country": "UK"},
    {"name": "Leicester", "lat": 52.6369, "lon": -1.1398, "country": "UK"},
    {"name": "Oxford", "lat": 51.7520, "lon": -1.2577, "country": "UK"},
    {"name": "Cambridge", "lat": 52.2053, "lon": 0.1218, "country": "UK"},
]

#Weather fetching function
def get_weather_condition(city_name):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city_name}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()['current']['condition']['text'].lower()
    except Exception as e:
        print(f" ! Weather fetch failed for {city_name}: {e}")
        return "unknown"
    
# Use Ollama to generate an emotional prompt based on weather
# Use Ollama to generate an emotional prompt based on weather
def get_emotional_prompt(city, weather_type):
    print("Generating prompt using Ollama... (this may take a few seconds)")
    try:
        emotional_contexts = {
            "rainy": "sad, blue, melancholic",
            "cloudy": "cold, thoughtful, introspective",
            "sunny": "happy, warm, joyful",
            "hail": "strange, chaotic, psychedelic",
            "snowy": "magical, peaceful, wonderful"
        }

        mood = emotional_contexts.get(weather_type.lower(), "neutral, abstract")

        prompt = (
            f"You are an emotional AI art critic. Based on the weather type '{weather_type}' in {city['name']}, "
            f"which feels {mood}, describe a painting prompt in the impressionist style that captures this emotion. "
            "Limit your response to under 3 sentences."
        )
        prompt = (
            f"You are an emotional AI art critic. Given the current weather condition '{weather_type}' in {city['name']}, "
            "describe a painting prompt in the impressionist style that conveys the emotional essence of this scene. "
            "Keep it under 3 sentences."
        )
        print(" Running Ollama LLM inference...")
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        print(" Prompt generation complete.")
        return result.stdout.strip()
    except Exception as e:
        print(f"WARNING: Ollama prompt generation failed: {e}")
        return (
            f"A {weather_type} day in {city['name']} painted in the impressionist style of Claude Monet. "
            "Soft brushstrokes, misty tones, atmospheric feeling."
        )


# Image generation for gesture
def generate_image_for_gesture(gesture_label, city):
    try:
        print("Generating image for gesture '%s' in city '%s'..." % (gesture_label, city['name']))
        global generation_start_time, tracking_enabled, current_city, current_condition

        prompt = get_emotional_prompt(city, gesture_label)
        prompts = [prompt]
        print("Prompt: %s" % prompt)

        # Save the prompt so Streamlit can display it
        with open("generated_images/last_prompt.txt", "w", encoding="utf-8", errors="replace") as f: f.write(prompt)


        image = pipeline.generate(
            prompts=prompts,
            uncond_prompts=None,
            input_images=[],
            strength=0.9,
            do_cfg=True,
            cfg_scale=7.5,
            height=512,
            width=512,
            sampler="k_lms",
            n_inference_steps=34,
            seed=None,
            models=models,
            device='cpu',
            idle_device='cpu'
        )[0]

        image_pil = Image.fromarray(np.asarray(image))
        output_folder = "generated_images"
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.join(output_folder, f"{gesture_label}_{city['name']}.png")
        image_pil.save(filename)
        print(f"Generated image saved at: {filename}")

        generation_start_time = time.time()
        tracking_enabled = False
        current_city = city
        current_condition = weather_cache.get(city['name'], "unknown")

    except Exception as e:
        print(f"ERROR: Failed to generate image for {gesture_label} in {city['name']}: {e}")
        sys.exit(1)


def find_closest_matching_city(condition):
    global weather_cache, last_fetch_time
    if time.time() - last_fetch_time > CACHE_INTERVAL:
        update_weather_cache()
        
    matches = []
    for city in UK_CITIES:
        weather = weather_cache.get(city['name'], "unknown")
        if condition in weather:
            distance = geodesic((51.5074, -0.1278), (city['lat'], city['lon'])).kilometers #calculate lat and lon, got help from chatgpt in this section
            matches.append((city, weather, distance))
    return sorted(matches, key=lambda x: x[2])[0] if matches else (None, None, None)

#Stable Diffusion Model load (Only ONCE)
print("Loading Stable Diffusion models")
models = model_loader.preload_models('cpu')

print("Models loaded")

GESTURE_DELAY = 12  # seconds
gesture_start_time = None
gesture_ready = False
GENERATION_DELAY = 12  # seconds
generation_start_time = None
tracking_enabled = True

#LSTM Model Class
class GestureLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        
        super(GestureLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.dropout(h[:, -1, :])
        out = self.fc(out)
        return out

#Model loading 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #for when I had to work on it on my nvidia pc
model = GestureLSTM(input_size=30, hidden_size=256, num_classes=5).to(device)
model.load_state_dict(torch.load('Gesture/gesture_lstm_model.pth', map_location=device))
model.eval()

le = LabelEncoder()
le.classes_ = np.load('Gesture/gesture_label_classes.npy', allow_pickle=True)

#MediaPipe hands init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#Webcam init
cap = cv2.VideoCapture(0) #
sequence_buffer = deque(maxlen=30)
SELECTED_INDICES = [0, 4, 8, 12, 20]
FIXED_FEATURES_PER_FRAME = 30

#Weather cache setup in order for the model to not get too slow
current_city = None
current_condition = None
weather_cache = {}
last_fetch_time = 0
CACHE_INTERVAL = 600  #double the cache

def update_weather_cache():
    global weather_cache, last_fetch_time
    weather_cache = {}
    for city in UK_CITIES:
        condition = get_weather_condition(city['name'])
        weather_cache[city['name']] = condition
    last_fetch_time = time.time()
        #If gesture matches, find city and generate image
generated_once = set()
update_weather_cache()

print(" Real-time Gesture Weather Generator Started. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ STEP 2: Show overlay while generating, disable hand tracking
    if not tracking_enabled and current_city:
        try:
            cv2.putText(frame, f"Gesture: {gesture_label}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        except NameError:
            pass  # gesture_label may not be defined yet

    if not tracking_enabled:
        elapsed = time.time() - generation_start_time
        remaining = max(0, int(GENERATION_DELAY - elapsed))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f" Generating image... ({remaining}s)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if current_city:
            cv2.putText(frame, f"Gesture: {gesture_label}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"City: {current_city['name']}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Resume hand tracking after generation countdown
        if elapsed >= GENERATION_DELAY:
            tracking_enabled = True
            sequence_buffer.clear()
            gesture_start_time = None
            gesture_ready = False
            print("Check Generation complete. Ready for new gesture.")

    results = hands.process(frame_rgb)

    combined_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for idx in SELECTED_INDICES:
                lm = hand_landmarks.landmark[idx]
                combined_landmarks.extend([lm.x, lm.y, lm.z])

    # Pad or truncate to FIXED_FEATURES_PER_FRAME
    if len(combined_landmarks) < FIXED_FEATURES_PER_FRAME:
        combined_landmarks.extend([0.0] * (FIXED_FEATURES_PER_FRAME - len(combined_landmarks)))
    elif len(combined_landmarks) > FIXED_FEATURES_PER_FRAME:
        combined_landmarks = combined_landmarks[:FIXED_FEATURES_PER_FRAME]

    sequence_buffer.append(combined_landmarks)


    if len(sequence_buffer) >= 30:
        sample = np.array(sequence_buffer, dtype=np.float32)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sample_tensor)
            pred = torch.argmax(output, dim=1).item()
            gesture_label = le.inverse_transform([pred])[0]

            # Start timer on first detection of valid gesture
            if gesture_label in ["rainy", "sunny", "hail", "cloudy", "snowy"]:
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                    print(f" Detected '{gesture_label}' gesture - warming up...")

                elapsed = time.time() - gesture_start_time

                if not gesture_ready:
                    remaining = int(GESTURE_DELAY - elapsed)
                    if remaining > 0:
                        cv2.putText(frame, f"Starting in {remaining}s...",
                                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        print(f" Gesture '{gesture_label}' confirmed after {GESTURE_DELAY}s")
                        gesture_ready = True

                # Now allow generation ONLY after delay
                if gesture_ready:
                    gesture_key = f"{gesture_label}_{gesture_label}"
                    if gesture_key not in generated_once:
                        city, condition, _ = find_closest_matching_city(gesture_label)
                        if city:
                            generate_image_for_gesture(gesture_label, city)
                            generated_once.add(gesture_key)
                            sequence_buffer.clear()
                            gesture_start_time = None
                            gesture_ready = False
                            print(" Ready for next gesture...")
            else:
                # Reset if gesture is not one of the targets
                gesture_start_time = None
                gesture_ready = False


        #Display gesture
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #Overlay closest city and weather
    if current_city:
        cv2.putText(frame, f"Closest City: {current_city['name']} ({current_city['country']})",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Condition: {current_condition.capitalize()}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No matching city found.",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #  Show video feed Generation in 
    cv2.imshow("Gesture Controlled Weather Generator", frame)

    #  Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Clean up
cap.release()
cv2.destroyAllWindows()

# Keep OpenCV window open until user presses 'q' or closes it manually
while True:
    if cv2.getWindowProperty("Gesture Controlled Weather Generator", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
