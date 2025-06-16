Project for the MSC in Data Science & AI for Creative Industries, 24/25 Data Science in the Creative Industries - IU000134 - Felipe Lopez
# 🤖 ClimaSign – Gesture-Controlled Weather Art Generator

ClimaSign is a real-time, AI-powered interactive installation that uses hand gestures to generate AI art based on current UK weather conditions. Leveraging computer vision (MediaPipe), gesture recognition (LSTM), weather APIs, and image generation (Stable Diffusion + LLaMA3 via Ollama), ClimaSign brings together multiple AI systems in a single expressive tool.

Project Assets can be Accessed [HERE](https://artslondon-my.sharepoint.com/:f:/r/personal/f_lopezmantilla0520231_arts_ac_uk/Documents/DS_Project_Felipe_Lopez?csf=1&web=1&e=nOa3gW) 

## 🌟 Features

- Real-time gesture detection via webcam (OpenCV + MediaPipe)
- Gesture classification with custom-trained LSTM
- Weather data from cities across the UK
- Artistic prompt generation using LLaMA3 via Ollama
- High-quality image generation using Stable Diffusion
- Live UI via Streamlit with auto-refreshing panels
- Logs and prompt tracking
- Auto-matching cities based on gesture-triggered weather condition

---

## 🚀 How to Run

You can run ClimaSign in two ways: **with a Streamlit UI** or **as a direct OpenCV application**.

### ✅ Requirements

- Python 3.8+
- Conda or virtualenv recommended
- Webcam
- GPU recommended for fast generation (supports CPU fallback)

### 📁 1. Clone this repository

```bash
git https://github.com/felipelopezm1/ClimaSign.git
cd ClimaSign
```

### 📦 2. Set up the environment

```bash
conda create -n clima python=3.9
conda activate clima
pip install -r requirements.txt
```

### 🔑 3. Secrets / Dependencies

- Get your **WeatherAPI key** from [weatherapi.com](https://www.weatherapi.com/)
- Rename `.env.example` to `.env` and add your key:

```
WEATHER_API_KEY=your_key_here
```

- Download required model files (LSTM model, gesture labels, Stable Diffusion config, etc.) from the following shared folder:

[📥 Download Project Assets](https://artslondon-my.sharepoint.com/:f:/g/personal/f_lopezmantilla0520231_arts_ac_uk/Eg1scnTgBYVNjfd9Tyqr660BXcBE7zXLtZhlIGwAVoWCsw?e=xQJl0k)

Unzip and place contents in the correct folders:
```
Gesture/
generated_images/
```

### 🖼 4. Running with Streamlit UI

```bash
streamlit run streamlit_gesture_ui.py
```

This opens the interface in your browser. You can run the detection script and view live logs + latest generated artwork.

### 🧠 5. Running directly (OpenCV Only)

```bash
python prediction_art_form_v6.py
```

This launches the gesture detection and generation system directly in your OpenCV webcam window.

---

## 🧪 Output

- All generated images are saved to `generated_images/`
- Prompts used are logged to `generated_images/last_prompt.txt`
- Logs are written to `logs/output.txt`

---

## ⚙️ Configurable Variables

- Weather cache duration
- LSTM model path and input size
- Gesture trigger delay
- Stable Diffusion generation parameters (e.g., CFG scale, sampler)

These can be adjusted in `config.py` (to be extracted in cleanup step).

---

## 📜 License

MIT License. See `LICENSE` file.

---

## 🤝 Acknowledgements

This project was developed with the help of **OpenAI’s ChatGPT**, and Implements AI Models like **LLMs like LLaMA3** via **Ollama** and **Stable Diffusion Pytorth** By **Stability AI**.
Inspired by the convergence of **data science**, **real-time ML**, and **generative art**.

---

Enjoy using ClimaSign! 🎨🖐️🌦️
