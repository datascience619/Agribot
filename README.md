# Agribot


# 🌾 Agricultural Assistant App

A smart farming desktop application that empowers farmers and agriculturalists with intelligent tools to improve productivity and sustainability. Built using **Python**, the app combines **Machine Learning**, **Deep Learning**, **Computer Vision**, and **Natural Language Processing** into a simple **Tkinter GUI**.

🚀 Features

 ✅ Crop Recommendation

* Input soil parameters: **pH**, **Nitrogen**, **Phosphorus**, and **Potassium**.
* Recommends the most suitable crop using a trained **Random Forest Classifier**.

 🦠 Plant Disease Detection

* Upload leaf images of crops like **Tomato**, **Potato**, **Corn**, or **Pepper**.
* Predict plant diseases using a pre-trained **CNN model (Keras)**.
* Supports 18+ plant disease classes from the **PlantVillage dataset**.

💬 Chatbot Assistant

* Ask general farming queries such as:

  * Suitable crops based on soil.
  * Remedies for plant diseases.
  * Government schemes for farmers.
* Powered by **NLTK's rule-based chatbot engine**.

🧪 Tech Stack

Python
  TkinterGUI)
  Pandas, NumPy, **scikit-learn** (Crop Recommendation)
  TensorFlow/Keras (Plant Disease Detection)
  NLTK (Chatbot)
  OpenCV (Image Processing)

 📷 Screenshots

> 📌 Add relevant screenshots of the UI here showing:

* Crop input and prediction
* Image upload and disease prediction
* Chatbot interaction

📁 How to Run

1. Clone the repository:

   bash
   git clone https://github.com/datascience619/Agricultural-Assistant.git
   cd Agricultural-Assistant

2. Install required packages:

   bash
   pip install -r requirements.txt
   

3. Run the application:

   bash
   python main.py
  

> ⚠️ Ensure the pre-trained plant disease model (`.h5` file) is available in the correct directory.
📂 Dataset Sources

Crop Recommendation: Sample soil data & `Crop_recommendation.csv`
Plant Disease Detection: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

 🎯 Future Enhancements

* Integration with live weather API.
* Voice-based chatbot support.
* Android/mobile version for real-time farming support.


