# Facial Keypoints Detection with VGG16

This project implements a facial keypoint detection model using a pre-trained VGG16 architecture. The dataset contains images of human faces with keypoints (such as eyes, nose, and mouth) labeled for each image.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- pandas
- gdown
- PIL (Pillow)
- numpy
- tqdm
- matplotlib

## How to Run

1. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

2. Download and extract the dataset:
    ```python
    import gdown
    url = 'https://drive.google.com/uc?export=download&id=1GlA5NQMSImR51HSfXjylQVF_aKZbaRrO'
    output='/kaggle/working/data.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall(output, '/kaggle/working/dataset')
    ```

3. Train the model:
    ```python
    python train.py
    ```

4. Visualize results:
    ```python
    python visualize.py
    ```

## Model Architecture

- **Base model**: Pre-trained VGG16
- **Modifications**: Classifier changed to predict 68 keypoints (136 values: x and y coordinates).
- **Loss function**: MSELoss
- **Optimizer**: Adam

## Performance

The model trains over multiple epochs, and performance is evaluated based on training and validation loss.
Performance curves:
![image](https://github.com/user-attachments/assets/ff50209f-1118-4663-8145-6e7737fc33de)

## Visualization

Keypoints are plotted over test images to visualize predictions.

## Final Result
![image](https://github.com/user-attachments/assets/af8c7956-b9f6-40e8-a357-cf509aa45fc8)


