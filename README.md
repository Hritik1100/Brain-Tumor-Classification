### Brain Tumor Classification

This project aims to classify brain tumor images into four categories: glioma, meningioma, no tumor, and pituitary. The dataset consists of MRI scans provided in separate training and testing folders.

### Purpose

The primary objective of this project is to develop a deep learning model capable of accurately classifying brain tumors from MRI images. This can assist in early diagnosis and treatment planning, ultimately improving patient outcomes.
## Methodologies
### Data Preparation

    Data Extraction: Extract the dataset from a zip file to a temporary directory.
    Image Processing: Read and resize the images to a uniform size of 128x128 pixels, and convert them to grayscale.
    Data Augmentation: Apply data augmentation techniques to enhance the diversity of the training dataset and improve model generalization.

Model Development

    Convolutional Neural Network (CNN): A CNN model is designed with several convolutional layers followed by max-pooling layers to extract features from the images.
    Regularization: Dropout layers are used to prevent overfitting by randomly dropping units during training.
    Fully Connected Layers: Flatten the output from the convolutional layers and pass it through fully connected layers to make the final classification.
    Activation Functions: ReLU activation is used for intermediate layers, and softmax activation is used for the output layer to get probability distributions for the four categories.

Training and Evaluation

    Training: The model is trained using the Adam optimizer and categorical cross-entropy loss. The dataset is split into training and validation sets to monitor performance and avoid overfitting.
    Evaluation: Model performance is evaluated on the testing set, and accuracy metrics are recorded.

Libraries Used

    NumPy: For numerical operations and array manipulation.
    Pandas: For data manipulation and analysis.
    Seaborn and Matplotlib: For data visualization.
    OpenCV: For image processing tasks.
    scikit-learn: For data splitting and model evaluation.
    TensorFlow/Keras: For building and training the CNN model.

Results

The training history is visualized using loss and accuracy plots to analyze model performance over epochs. The trained model is then evaluated on the test dataset to determine its accuracy.

Conclusion

This project demonstrates the use of deep learning techniques for medical image classification. The developed CNN model shows promising results in classifying brain tumors, which can be further improved with more data and advanced techniques.
