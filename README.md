# Similar Images Remover Tool

## Description

This program is designed to find and remove similar-looking images in a folder containing a dataset of images collected from cameras. The goal is to optimize the dataset by removing duplicated or almost duplicated images that have minor differences and are considered non-essential for data collection and object recognition tasks.

The program uses provided functions for image comparison from `imaging.py` to assess the similarity between images. Specifically, it leverages the `preprocess_image_change_detection` function to preprocess the images and the `compare_frames_change_detection` function to compute a similarity score between a pair of images.

[comment]: <> (## Subsample Dataset)

[comment]: <> (For experimentation, we have provided a subsample of our dataset in the `dataset-candidates-ml.zip` file. The dataset filenames are formatted as follows: `c%camera_id%-%timestamp%.png`. The timestamps may be in two different formats.)

## Installation

To run the program, follow these steps:

1. Clone this GitHub repository to your local machine.
2. Ensure you have Python installed (version 3.6 or later).
3. Install the required libraries by running: `pip install -r requirements.txt`.

## Usage

1. Unzip your dataset (Eg. `dataset.zip`) into a folder.
2. Execute the program by running the following command in the terminal:

```python similar_images_remover.py --folder_path "/path/to/dataset_folder"```

Replace `"/path/to/dataset_folder"` with the path to the folder containing your dataset.

## Hyperparameters
| Hyperparameter               | Description                                                                                   | Default Value   |
|------------------------------|-----------------------------------------------------------------------------------------------|-----------------|
| `--threshold`                | Adjusts the similarity threshold for image removal. (Lower values result in stricter removal.) | 0.85             |
| `--min_contour_area`         | Minimum contour area for image comparison. (Lower values result in stricter removal.)        | 500               |
| `--gaussian_blur_radius`     | A list of Gaussian blur radii for image preprocessing to remove high frequency features.     | ["None"]            |
| `--black_mask`               | Percentage values (left, top, right, bottom) for the black mask applied to image borders.    | (0, 15, 0, 0)    |
| `--frame_change_thresh`      | Threshold to convert grayscale images into binary.                                           | 25              |
| `--resize_shape`      | Size to reshape the images.                                           | (200,200)              |


## Features

- **Automated Similarity Detection:** The tool employs classical image comparison techniques to automatically detect and identify similar-looking images within the dataset. It uses a combination of preprocessing and contour analysis to ensure accurate and reliable similarity detection.


- **Adjustable Similarity Threshold:** Users have the flexibility to fine-tune the similarity threshold (`--threshold`) to control the strictness of image removal. This allows for customization based on the specific requirements of the dataset and object recognition tasks.


- **Visual Preprocessing Analysis:** The tool provides a built-in functionality to visualize the preprocessing steps applied to the images (using `visualize_preprocess_image`). Users can experiment with different Gaussian blur radii (`--gaussian_blur_radius`) and black mask percentages (`--black_mask`) to understand their impact on image similarity scores. The Gaussian blur radius list determines the radius of the blurring kernel used for smoothing the images, while the black mask percentages control the size of the black mask applied to the image borders.


- **Smart Data Optimization:** By removing duplicated or nearly identical images, the Similar Images Removal Tool efficiently optimizes the dataset. This optimization leads to reduced storage requirements, faster training times, and improved model generalization.

