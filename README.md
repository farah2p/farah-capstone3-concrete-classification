# Concrete Crack Detection
## Project Description
This repository contains the code and resources for a machine learning project focused on concrete crack detection. The project aims to develop an image classification model to classify concrete images into two categories: "With Cracks" and "Without Cracks". By accurately identifying concrete cracks, this AI-based solution can help prevent potential risks to building safety and durability. The project utilizes a dataset obtained from Mendeley and follows a machine learning workflow from problem formulation to model deployment.
## Dataset
The dataset used for this project can be found at the following link: https://data.mendeley.com/datasets/5y9wdsg2zt/2
## Project Functionality
- Performs image classification to identify concrete cracks.
- Addresses various types of concrete cracks, including hairline cracks, shrinkage cracks, settlement cracks, and structural cracks.
- Ensures model accuracy, avoiding overfitting.
- Applies transfer learning to leverage pre-trained models.
- Presents a well-structured and presentable project on GitHub.
- Deploys the model to make predictions on test data.
## Challenges and Solutions
### Challenge: 
Dataset collection and preprocessing.
### Solution:
Obtained the concrete crack dataset from Mendeley and performed necessary preprocessing steps to prepare the data for model training.
### Challenge: 
Model accuracy and overfitting prevention.
### Solution:
Implemented transfer learning and regularization techniques to achieve training and validation accuracy of more than 90% while preventing overfitting.
### Future Implementations
- Implement a web application to allow users to upload images and classify concrete cracks in real-time.
- Enhance the model's performance by exploring advanced deep learning architectures and techniques.
- Develop a mobile application to enable on-the-go concrete crack identification.
## Installation and Usage
### 1. Clone the repository to your local machine using the following command:
```shell
git clone https://github.com/farah2p/farah-capstone3-concrete-classification.git
```
### 2. Before running the code, ensure that you have the following dependencies installed:
- TensorFlow
- Pandas 1.5.3
- Matplotlib
- Tensorboard 2.12.3

Install the required dependencies by running the following command:
```shell
pip install tensorflow==2.12.0
pip install numpy==1.24.2
pip install matplotlib==3.7.1
pip install pandas==1.5.3
pip install tensorboard===2.12.3
```
### 3. Download the dataset from https://data.mendeley.com/datasets/5y9wdsg2zt/2 and place it in the project directory.
### 4. Open the Jupyter Notebook or Python script containing the code.
### 5. Run the code cells or execute the script to perform the following tasks:
- Preprocess the concrete images for training and evaluation.
- Develop the image classification model using deep learning techniques.
- Train the model, ensuring accuracy and preventing overfitting.
- Evaluate the model using validation data.
- Deploy the trained model to make predictions on the test dataset.
### 6. Use Tensorboard to visualize the model graph and training progress by running the following command in the project directory:
```shell
tensorboard --logdir tensorboard_logs/capstone3
```
Access Tensorboard in your web browser using the provided URL.
### 7. The trained model will be saved in the h5 and pkl format as capstone3_model.h5 and capstone3_model.pkl, respectively
## Workflow
The machine learning workflow for this project can be divided into the following stages:
### 1) Problem Formulation
#### Objective
The objective of this project is to develop a machine learning model that can accurately classify images of concrete as either cracked or non-cracked. By achieving high accuracy in classifying concrete images, the model aims to assist in the identification of potentially hazardous cracks in concrete structures, promoting safety and durability.
#### Problem Statement
- Concrete cracks pose significant risks to the structural integrity of buildings and infrastructure. Identifying and addressing these cracks in a timely manner is crucial for preventing accidents, minimizing damage, and ensuring the longevity of structures. However, manual inspection and identification of concrete cracks can be time-consuming and error-prone.
- The problem addressed in this project is to automate the process of concrete crack detection using machine learning techniques. By leveraging the power of image classification algorithms, the developed model will be able to classify concrete images as cracked or non-cracked with a high degree of accuracy. This will enable faster and more reliable identification of concrete cracks, facilitating prompt maintenance and repairs.
- The machine learning model will be trained on a dataset of concrete images labeled as cracked or non-cracked. The model will learn the distinguishing features and patterns that differentiate cracked concrete from non-cracked concrete. It will then be able to generalize this knowledge to classify new, unseen concrete images accurately.
- The ultimate goal is to create a reliable and efficient tool that can assist engineers, construction professionals, and inspectors in identifying concrete cracks, leading to improved safety standards and better maintenance practices for concrete structures.
- By completing this project successfully, the model developed will have the potential to save lives, prevent accidents, and contribute to the long-term durability of buildings and infrastructure.
### 2) Data Preparation
#### Preprocessing:
- Load the concrete crack dataset, which consists of images labeled as cracked or non-cracked. The dataset is organized into the Positive and Negative folders, respectively.
- Perform any necessary preprocessing steps, such as resizing the images to a consistent resolution.
#### Splitting into Training and Validation Sets:
- To divide the dataset, you can use the following approach:
  - Split the dataset into training, validation, and test sets. A typical split could be 70% for training, 20% for validation, and 10% for testing.
  - Determine the size of each set by calculating the respective percentages based on the total size of the dataset. For example, if the dataset contains 100 samples:
  Training set size: 70% of 100 = 70 samples
  Validation set size: 20% of 100 = 20 samples
  Test set size: 10% of 100 = 10 samples
- You can adjust these percentages according to your specific requirements.
- Ensure that the data distribution is balanced between the cracked and non-cracked classes in both the training and validation sets. This helps prevent bias in the model's performance.

NOTES: It's important to note that the specific preprocessing and augmentation techniques used may vary depending on the characteristics of the concrete crack dataset and the requirements of the model. Experimentation and exploration of the data are essential to determine the most effective preprocessing steps and augmentation strategies for this project.
### 3) Model Development
Train an image classification model using deep learning techniques. Transfer learning was applied to improve performance. In the model development stage of the concrete crack classification project, train an image classification model using deep learning techniques. Transfer learning can be employed to enhance the model's performance by leveraging pre-trained models. Here's an outline of the steps involved:
Selecting a Deep Learning Framework:
- Choose a deep learning framework such as TensorFlow, PyTorch, or Keras that supports building and training convolutional neural networks (CNNs).
- Install the necessary libraries and set up the development environment.
Loading Pre-trained Models:
- Explore and select a pre-trained CNN model that has been trained on a large-scale image dataset and imagenet was used in this dataset.
- Load the pre-trained model's architecture and weights into your development environment.
- Common pre-trained models used in image classification include VGG, ResNet, Inception, and MobileNet.
Modifying the Model:
- Adapt the pre-trained model's architecture to suit the concrete crack classification task.
- Add new layers to the pre-trained model to adjust the number of output classes to two (cracked and non-cracked).
- Ensure that the dimensions of the input images match the input size expected by the pre-trained model.
- The model that has being used is MobileNetV2, and the architecture is adapted for the concrete crack classification task.
Training the Model:
- Initialize the model with the modified architecture and compile it with an appropriate loss function and categorical cross-entropy has being used and Adam as the optimizer.
- Train the model using the prepared training dataset.
- Monitor the training process, including metrics such as accuracy and loss, to assess the model's performance.
- Experiment with hyperparameters, such as learning rate and batch size, to optimize the model's training.
Evaluating the Model:
- After training, evaluate the model's performance using the prepared validation dataset.
- Calculate metrics such as accuracy, precision, recall, and F1 score to assess the model's effectiveness in classifying concrete cracks.
Fine-tuning (Optional):
- If the model's performance is not satisfactory, consider fine-tuning the model.
- Fine-tuning involves unfreezing additional layers of the pre-trained model and training them with a lower learning rate to further adapt to the concrete crack classification task.
- Fine-tuning can help the model capture more specific features relevant to concrete cracks.

NOTES: By employing deep learning techniques and leveraging pre-trained models, you can benefit from the transfer of learned features and accelerate the model's training and performance. Remember to experiment with different architectures, hyperparameters, and fine-tuning strategies to achieve the desired accuracy and generalization capability for concrete crack classification.
### 4) Model Deployment
In the model deployment stage of the concrete crack classification project, deploy the trained model and utilize it to make predictions on the test data. Here's an overview of the steps involved:
Saving the Trained Model:
- After training and evaluating the model, save the trained model's architecture and weights to disk.
- This ensures that the model can be loaded and used for predictions without retraining.
Loading the Trained Model:
- In the deployment environment, load the saved model architecture and weights into memory.
- This step typically involves using the same deep learning framework and libraries used during model development.
Preparing the Test Data:
- Preprocess the test data in a similar manner to the training and validation data.
- Ensure that the test data is formatted correctly, such as resizing images to the same dimensions expected by the model.
Making Predictions:
- Use the loaded model to make predictions on the preprocessed test data.
- Feed the test data through the model, and obtain the predicted classes or probabilities for each input.
- Convert the model's output into meaningful predictions (e.g., determining whether the concrete is cracked or non-cracked).
Performance Evaluation:
- Compare the model's predictions with the ground truth labels of the test data to assess its performance.
- Calculate evaluation metrics such as accuracy, precision, recall, and F1 score to measure the model's effectiveness in concrete crack classification.
Handling New Images:
- Extend the deployment functionality to accept new images for classification.
- Ensure that the new images undergo the same preprocessing steps as the training and test data.
- Utilize the loaded model to make predictions on the new images, providing real-time concrete crack detection.

NOTES: By deploying the trained model and utilizing it for predictions on the test data, you can evaluate its performance on unseen samples and verify its effectiveness in concrete crack classification. Additionally, extending the deployment to handle new images allows for practical and real-world usage of the model to aid in concrete crack detection.
## Criteria
To successfully complete this project, the following criteria should be met:
- Achieve a training and validation accuracy of more than 90%.
- Ensure the model is not overfitting by monitoring the training and validation loss.
- Apply transfer learning techniques to leverage pre-trained models.
- Use the provided dataset mentioned in the README. Do not upload the data onto GitHub.
- Present the entire project in a presentable manner, adhering to good coding practices:
- Use proper variable naming conventions.
- Write concise comments to improve code readability.
## Credits
The dataset used in this project is sourced from:
https://data.mendeley.com/datasets/5y9wdsg2zt/2
## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.
