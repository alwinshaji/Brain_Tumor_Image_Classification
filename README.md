# 🧠 Brain Tumor Image Classification

This project is a **Brain Tumor Classification App** built with **TensorFlow**, **Keras**, and **Streamlit**.  
The model classifies brain MRI images into four categories:

- **Glioma**  
- **Meningioma**  
- **No Tumor**  
- **Pituitary**

The model is trained on MRI datasets and deployed using **Streamlit Cloud**.

---

## 🚀 Live App
Check out the live app here:  
**[Brain Tumor Classification - Streamlit App](https://braintumorimageclassification.streamlit.app)**

---

## 📂 Repository Structure

├── app.py # Streamlit application
├── Brain_Tumor_MRI_Image_Classification.ipynb # Jupyter notebook for training
├── requirements.txt # Dependencies for the app
├── runtime.txt # Python version for deployment
└── README.md # Project documentation


---

## 🧩 Model
The trained model (`final_trained_model.h5`) is hosted on Google Drive due to file size limitations.

**Download link:**  
[final_trained_model.h5 (Google Drive)](https://drive.google.com/file/d/1otdiwo82KkKWNMebhTI7BUDQf55-mB4H/view?usp=drive_link)

The `app.py` file automatically downloads the model from the above link if it is not present locally.

---

## 📊 Dataset
The dataset used for training and evaluation can be accessed here:  
[Brain MRI Dataset (Google Drive)](https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju?usp=sharing)

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alwinshaji/Brain_Tumor_Image_Classification.git
   cd Brain_Tumor_Image_Classification

2. **Create a virtual environment and activate it:**
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install dependencies:**
   pip install -r requirements.txt

4. Run the Streamlit app:
   streamlit run app.py

5. Open your browser and navigate to http://localhost:8501.

---

📒 Notebook

The Brain_Tumor_MRI_Image_Classification.ipynb file contains:

Data preprocessing
Model training (CNN)
Evaluation and saving the final model (final_trained_model.h5)

---

🛠 Technologies Used

Python 3.11
TensorFlow 2.18.0
Keras 3.8.0
Streamlit
NumPy, Pandas, Matplotlib, Seaborn
Pillow

---

✨ Features

Upload an MRI image to get tumor type prediction.
Displays prediction confidence for all classes.
Uses a trained CNN model for inference.
Deployed on Streamlit Cloud.

---

📧 Contact
If you have any issues or suggestions, feel free to reach out via GitHub Issues.











   


