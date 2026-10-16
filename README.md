# Plant Leaf Disease Detection

A Streamlit web app that identifies plant leaf diseases from a photo using a TensorFlow/Keras model (EfficientNetB0) trained on the PlantVillage dataset (15 classes).

## Run locally (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

## Train the model
Place the PlantVillage class folders inside `dataset/`, then run:
python training\train_model.py

## Limitations
- The model only knows 15 classes (pepper, potato, tomato) and will always pick one, even for non-leaf images.
- Trained on lab-style images, so field photos may give lower accuracy.
- Results are guidance only. Consult a local agriculture expert before treating crops