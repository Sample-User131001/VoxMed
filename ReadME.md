## VoxMed: One-Step Respiratory Disease Classifier using Digital Stethoscope Sounds - Readme

We introduce VoxMed, a simple UI system that detects respiratory diseases using digital stethoscope sounds. Our goal is to overcome the challenges of traditional respiratory diagnostics, which typically involve long waits in crowded clinical settings . With VoxMed, healthcare practitioners can efficiently respond to patients' medical needs based on fast respiratory health assessments.
Past studies have shown that machine learning algorithms can potentially evaluate medical audio data to identify illnesses. VoxMed elaborates on the previous works to develop a one-step interface that enables doctors to record patient stethoscope sounds and support seamless diagnosis and analysis of patients. VoxMed accomplishes this by employing an Audio Spectrogram Transformer(AST) to extract features and a 1-D Convolutional Neural Network (CNN) architecture for classifying diseases, as shown in Figure 1. 
We test VoxMed's performance using the ICBHI challenge dataset, featuring stethoscope sounds gathered from patients in Greece and Portugal. Our findings demonstrate that VoxMed reliably identifies respiratory diseases such as COPD, potentially enhancing patient care.
Our findings illustrate the platform's ability to reliably identify respiratory disorders, underlining its potential as a useful tool for respiratory disease diagnosis and therapy in clinical practice.

**Features:**

- Uploads a digital stethoscope audio file (WAV format).
- Extracts features from the audio using a pre-trained Audio Set Transfer (AST) model.
- Predicts the most likely respiratory disease based on the extracted features using a deep learning model.
- Displays informative messages and relevant images based on the prediction.

**Requirements:**

- Python 3.x
- Streamlit (`pip install streamlit`)
- TensorFlow (`pip install tensorflow`)
- PyTorch (`pip install torch`)
- torchaudio (`pip install torchaudio`)
- transformers (`pip install transformers`)
- Pillow (`pip install Pillow`)

**Instructions:**

1. Download the pre-trained AST model or Import it From the Hugging Face Website and disease classification model:
    - Download the AST model files (e.g., `pytorch_model.bin`) from [https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (replace with the actual download URL). Place them in a directory.
    - Download the disease classification model (`Model.h5`) and place it in the same directory as the AST model files.
2. Update file paths in the code:
    - Unzip the Assets zip file
    - Modify the following paths to reflect your actual locations:
        - `'C:\\Users\\UserName\\Desktop\\RESPIRATORY DISEASE CLASSIFIER\\Model.h5'` (path to your disease classification model)
        - `'C:\\Users\\UserName\\Desktop\\RESPIRATORY DISEASE CLASSIFIER\\Assets\\Healthy.gif'` (path to the healthy image)
        - `'C:\\Users\\UserName\\Desktop\\RESPIRATORY DISEASE CLASSIFIER\\Assets\\RESPIRATORY-DISORDERS-.jpg'` (path to the generic respiratory issues image)
        - `'C:\\Users\\UserName\\Desktop\\RESPIRATORY DISEASE CLASSIFIER\\Assets\\COPD.png'` (path to the COPD info image )
3. Run the application:
    - Open a terminal and navigate to the directory containing the script (`APP.py`).
    - Run the script using `streamlit run "C:\Users\UserName\Desktop\RESPIRATORY DISEASE CLASSIFIER\Assets\APP.py"`.
4. Use the application:
    - Upload an audio file from your digital stethoscope.
    - The application will display the predicted disease, relevant information, and images.
    - For COPD prediction, an additional information button can be clicked to display a detailed explanation.
