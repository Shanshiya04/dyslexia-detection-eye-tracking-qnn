# dyslexia-detection-eye-tracking-qnn
Final year project on multimodal dyslexia detection using eye-tracking data. The system integrates machine learning (SVM, Logistic Regression), deep learning (ResNet18), and Quantum Neural Networks (QNN) with a fusion-based approach for subject-level classification.
Got it 👍 — you want the README to look more like a **clean thesis-style paragraph format**, not bullet-heavy. Here’s a **refined, professional version** with paragraph flow under each section.





## 📌 **Prediction of Dyslexia Through Eye Movement Patterns Using Multimodal AI and Quantum Neural Networks**



## 📖 **Overview**

This project presents a multimodal artificial intelligence framework for automated dyslexia detection using eye-tracking data. The proposed system integrates both numerical gaze features and fixation-based visual representations to analyze reading behavior and classify subjects as dyslexic or non-dyslexic. By combining multiple learning paradigms, the framework provides a robust and scalable solution for objective dyslexia screening.

 

## 🎯 **Objective**

The primary objective of this project is to develop an automated and reliable dyslexia detection system that overcomes the limitations of traditional diagnostic methods. The system aims to leverage eye-tracking data to provide objective analysis of reading behavior. Additionally, the project focuses on integrating machine learning, deep learning, and quantum neural network approaches to improve classification performance and ensure robust predictions through multimodal fusion.

 
## 🧠 **Methodology**

The proposed system follows a dual-branch architecture that processes eye-tracking data through numerical and image-based pipelines. In the numerical branch, statistical gaze features such as fixation duration, saccade amplitude, and regression behavior are extracted and used as inputs for classical machine learning models, including Logistic Regression and Support Vector Machine (SVM). In the image-based branch, fixation-based visual representations are analyzed using a deep learning model (ResNet18) to extract spatial features. These features are further processed using a Quantum Neural Network (QNN) to capture complex non-linear relationships. The predictions from all models are combined using a majority voting fusion strategy to produce the final classification.

 

## 📊 **Dataset**

The system is evaluated using the ETDD70 Eye-Tracking Dyslexia Dataset, which consists of eye movement recordings from 70 children, including 35 dyslexic and 35 non-dyslexic subjects. The dataset includes three types of reading tasks: syllable reading, meaningful text reading, and pseudo-text reading. These tasks are designed to capture different aspects of reading behavior and provide both temporal and spatial information for analysis.

 

## 📈 **Results**

The experimental results demonstrate that classical machine learning models, namely SVM and Logistic Regression, achieve the highest classification accuracy of 92.86%. The deep learning model (ResNet18) and the Quantum Neural Network achieve an accuracy of 85.71%. The fusion model, based on majority voting, achieves an overall accuracy of 92.86% while improving robustness and reducing the impact of individual model errors. The results highlight the effectiveness of numerical gaze features and the complementary role of image-based and quantum approaches.
| Model               | Accuracy   |
| ------------------- | ---------- |
| SVM                 | 92.86%     |
| Logistic Regression | 92.86%     |
| ResNet18            | 85.71%     |
| QNN                 | 85.71%     |
| Fusion Model        | **92.86%** |
 

## ⚙️ **Technologies Used**

This project is implemented using Python and incorporates various libraries and frameworks, including Scikit-learn for machine learning, PyTorch for deep learning, and PennyLane for quantum neural network implementation. Additional libraries such as NumPy, Pandas, and Matplotlib are used for data processing and visualization.


## 🚀 **How to Run**

To run the project, clone the repository to your local system, install the required dependencies, and execute the main script. The system will process the input data, train the models, and generate classification results based on the implemented multimodal framework.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
python main.py
```

---

## 📁 **Project Structure**

The project is organized into multiple modules, including data preprocessing, feature extraction, model implementation, and result evaluation. Separate directories are maintained for machine learning models, deep learning models, and quantum neural network components to ensure modularity and ease of development.

---

## 🎓 **Academic Context**

This work is developed as a final year undergraduate project in the domain of machine learning, deep learning, and quantum computing. The application focuses on dyslexia detection using eye-tracking data, providing an innovative approach to cognitive disorder analysis.

---

## 🔬 **Key Contributions**

The key contribution of this project lies in the development of a unified multimodal framework that integrates classical machine learning, deep learning, and quantum neural network approaches. The system captures both temporal and spatial aspects of reading behavior and introduces a fusion-based strategy for improved classification robustness. Additionally, the use of subject-level evaluation ensures realistic and unbiased performance assessment.


## 👨‍💻 **Author**

Shanshiya E
Srilekha B S
Rudhuvarshan R


## 📜 **License**

This project is licensed under the MIT License.
 
