import streamlit as st

from tensorflow.keras.models import load_model
from PIL import Image
import time
import numpy as np

from transformers import AutoProcessor, ASTModel
import torchaudio
import torch

def stream_data(value1,path,value2):
    """
    Streams text data with an optional pause between words and displays an image.

    Args:
        value1 (str): The first text string to stream.
        path (str): The path to the image file.
        value2 (str): The second text string to stream.

    Yields:
        Generator[str, None, None]: A generator that yields each word from the provided strings.
    """
    for word in value1.split(" "):
        yield word + " "
        time.sleep(0.02)

    image = Image.open(path)
    st.image(image)
    for word in value2.split(" "):
        yield word + " "
        time.sleep(0.02)


disease_classes = ['COPD','Healthy', 'Other']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")# audio file is decoded on the fly
model.to(device)
def extract_features(path, device):
    """
    Extracts features from an audio file using a pre-trained Audio Set Transfer (AST) model.

    Args:
        path (str): Path to the audio file.
        device (torch.device): Device to use for computations (CPU or GPU).

    Returns:
        np.ndarray: A 1D NumPy array containing the extracted features (last hidden state of the model).

    Raises:
        ValueError: If the audio file cannot be loaded or processed.
    """
    sample_rate = 16000
    array, fs = torchaudio.load(path)
    array  = np.array(array)
    array = np.mean(array, axis = 0)
    input = processor(array.squeeze(), sampling_rate= sample_rate, return_tensors="pt")
    input=input.to(device)
    with torch.no_grad():
       outputs = model(**input)
    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).to("cpu").numpy()
    return last_hidden_states

st.title('One-Step Respiratory Disease Classifier using Digital Stethoscope Sound')


uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

if uploaded_file is not None:
    features = extract_features(uploaded_file,device)
    features=np.array(features)
    features = features.reshape((1, 768, 1))    
    model = load_model('C:\\Users\\UserName\\Desktop\\Model.h5')
    y_pred = model.predict(features)
    print(y_pred)
    predicted_index = np.argmax(y_pred,axis=1)
    print(predicted_index)
    predicted_disease = disease_classes[predicted_index[0]]
    
    if(predicted_disease=='Healthy'):
        image = Image.open('C:\\Users\\UserName\\Desktop\\Healthy.gif')
        st.image(image)
        st.write(f'Disease: No Disease')
        st.write('Cause:', 'No Infection')
        st.write('Remedies:', '''Nothing Required''')
        st.header('Possible Disease 🦠',divider='violet')
        st.subheader("Nothing to worry about, you are healthy! 🎉🎉🎉")
        st.header('Causes ❓',divider='violet')
        st.subheader("No Infection, No Disease, No Problem! 🎉🎉🎉")
        st.header('Remedies 💉',divider='violet')
        st.subheader("Nothing Required, You are Healthy! 🎉🎉🎉")

        st.header('Extra Information 💉',divider='violet')
        st.subheader("Nothing Required, You are Healthy! 🎉🎉🎉")
    elif(predicted_disease=='Other'):
        image = Image.open('C:\\Users\\UserName\\Desktop\\RESPIRATORY-DISORDERS-.jpg')
        st.image(image)
        st.header('Possible Disease 🦠',divider='violet')
        st.subheader("URTI, Bronchitis, LRTI, Pneumonia, Asthma")
        st.header('Causes ❓',divider='violet')
        st.subheader("_Asthma_ is often set off by allergies, infections, or environmental pollution, leading to airway inflammation and spasms. _Chronic Bronchitis_, while a type of COPD, deserves separate mention as it can result from long-term exposure to irritants like smoke or dust, causing persistent coughing. _Emphysema_, another COPD variant, arises primarily from smoking, which impairs the lungs’ alveoli. _Acute Bronchitis_ is typically a short-lived airway infection, usually viral in origin. Cystic Fibrosis, a genetic disorder, disrupts the body’s salt and water balance, creating thick mucus in the lungs. _Pneumonia_ infects the alveoli and can be bacterial or viral, including strains like the one responsible for COVID-19. _Tuberculosis_ is a bacterial lung infection with symptoms akin to pneumonia. _PulmonaryEdema_ involves fluid accumulation in the lung’s air sacs, often due to heart failure or lung injury. Lung Cancer may develop in any lung area, frequently linked to smoking or pollutants like radon and asbestos. Lastly, Acute Respiratory Distress Syndrome (ARDS) is a grave, rapid-onset lung injury from severe illnesses, including COVID-19. Each of these conditions highlights the lungs’ vulnerability to a spectrum of harmful factors.")
        st.header('Remedies 💉',divider='violet')
        st.subheader("Supportive care, Nasal suctioning, Oxygen therapy, Can be self-healing, IV fluids and Sodium chloride")

        st.header('Extra Information 💉',divider='violet')
        st.subheader('fast breathing, shortness of breath, wheezing, difficulty breathing, or shallow breathing Whole body: dehydration, fever, loss of appetite, or malaise Also common: coughing or nasal congestion')
    elif(predicted_disease=='COPD'):
        image = Image.open('C:\\Users\\User Name\\Desktop\\RESPIRATORY-DISORDERS-.jpg')
        st.image(image)
        _COPD_ = """
        A group of lung diseases that block airflow and make it difficult to breathe.
        Emphysema and chronic bronchitis are the most common conditions that make up COPD. Damage to the lungs from COPD can't be reversed.
        Symptoms include shortness of breath, wheezing or a chronic cough.
        Rescue inhalers and inhaled or oral steroids can help control symptoms and minimize further damage.
        """

        _INFO_='''Very common\n
        More than 10 million cases per year (India)\n
        ⚕️ Treatment can help, but this condition can't be cured\n
        📒 Requires a medical diagnosis\n
        🧪 Lab tests or imaging often required\n
        ⌛ Chronic: can last for years or be lifelong\n'''

        st.header('Disease 🦠',divider='violet')
        st.subheader("Chronic Obstructive Pulmonary Disease")
        st.header('Causes ❓',divider='violet')
        st.subheader("Smoking is the main cause of COPD and is thought to be responsible for around 9 in every 10 cases. The harmful chemicals in smoke can damage the lining of the lungs and airways.")
        st.header('Remedies 💉',divider='violet')
        st.subheader("Treatment is self care and bronchodilators to prevent flare ups and manage symptoms. Rescue inhalers and inhaled or oral steroids can help control symptoms and minimize further damage.",
        "Please Consult with Primary Care Provider")
        if st.button("More"):
            path='C:\\Users\\UserName\\Desktop\\COPD.png'
            st.write_stream(stream_data(_COPD_,path,_INFO_))
