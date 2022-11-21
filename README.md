# Creating a Lemon Quality Classifier with CNN.

## Dataset
I used the Lemon Dataset by Yusuf Emir.
https://github.com/robotduinom/lemon_dataset

### Dataset Preview
Bad Quality Lemon

![bad_quality_214](https://user-images.githubusercontent.com/111753936/202937348-eebd5e36-8411-49b3-b5aa-a4664e7bc6a9.jpg)

Good Quality Lemon

![good_quality_297](https://user-images.githubusercontent.com/111753936/202937389-4033b90a-38b4-4dc2-95ef-5165dfe9204e.jpg)

Empty Background

![empty_background_21](https://user-images.githubusercontent.com/111753936/202937401-b8639fdd-7adc-4260-a87f-b56f5db05631.jpg)

## The CNN Model
Consists of 3 convolutional layers with kernel size of 5, and 2 dense layers of size 64.
I am also using Cross Entropy Loss and Adam optimizer with learning rate of 0.0001.

### Model Results

Trained the model on CUDA with 30 epochs and around 2500 training images.
After 30 epochs, the model reached 99.27% accuracy and a loss of 0.5598. 
