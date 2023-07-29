import os
import random
import shutil
from datetime import datetime

import argparse
import cv2
import matplotlib.pyplot as plt
from imaging import preprocess_image_change_detection, compare_frames_change_detection

class SimilarImagesRemover:
    def __init__(self, folder_path, threshold=0.85, min_contour_area=500, gaussian_blur_radius_list=None,
                 black_mask=(0, 15, 0, 0), frame_change_thresh=25, resize_shape=(200, 200)):
        # Path to dataset images
        self.folder_path = folder_path
        # Create a new directory to store the removed images separately
        current_dir = os.getcwd()
        self.removed_images_path = os.path.join(current_dir, 'removed_images')
        if not os.path.exists(self.removed_images_path):
            os.makedirs(self.removed_images_path)

        # Initialize hyperparameters
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.gaussian_blur_radius_list = gaussian_blur_radius_list
        self.black_mask = black_mask
        self.frame_change_thresh = frame_change_thresh
        self.resize_shape = resize_shape

        # Dictionary to store file names and their corresponding similar looking images
        self.similar_images = {}
        self.num_images_removed = 0

    '''
    This function fetches and dumps the hyperparameter used in a particular experimental run and the dictionary having 
    all the file names along with their corresponding similar looking images with their similarity scores in a text file.
    '''
    def _write_to_logger(self):
        similar_images_with_hyperparameters = {
            "hyperparameters": {
                "threshold": self.threshold,
                "min_contour_area": self.min_contour_area,
                "gaussian_blur_radius_list": self.gaussian_blur_radius_list,
                "black_mask": self.black_mask
            },
            "similar_images": self.similar_images
        }

        # Write the similar_images_with_hyperparameters to output file
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = f"similar_images_{date_time_str}.txt"
        with open(output_file_name, 'wt') as output_file:
            output_file.write(str(similar_images_with_hyperparameters))

    '''
    This function processes each image by resizing it to a common shape and preprocessing it with Gaussian blur and black
    masking as specified in the class hyperparameters. It then calculates a hash for each image, representing its
    contents. The calculated hashes are stored in a dictionary with corresponding filenames for later similarity
    detection.
    
    Returns:
        dict: A dictionary containing image filenames as keys and their corresponding image hashes as values.
    '''
    def _get_image_hashes(self):
        # Create a dictionary to store image hashes and their filenames
        image_hashes = {}

        # Get a list of all image files in the folder
        image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Compare each image with others in the folder
        for file_name in image_files:
            file_path = os.path.join(self.folder_path, file_name)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Error: Failed to read image '{file_name}'. Skipping...")
                continue

            # Resize the image to a common size
            image = cv2.resize(image, self.resize_shape)
            # Preprocess the image to get the hash
            image_hash = preprocess_image_change_detection(image, self.gaussian_blur_radius_list, self.black_mask)
            image_hashes[file_name] = image_hash

        return image_hashes

    '''
    Compares image hashes to find similar images in the dataset. For each image, it computes its similarity with other
    images in the dataset using the `compare_frames_change_detection` function. The similarity score is normalized and
    images above the specified threshold are considered similar. Similar image filenames and their similarity scores 
    are stored in the self.similar_images dictionary.
    '''
    def _find_similar_images(self, image_hashes):
        # Compare image hashes to find similar images
        for file_name, image_hash in image_hashes.items():
            # Keep a list of all similar images for each file_name
            self.similar_images[file_name] = []
            for other_file, other_hash in image_hashes.items():
                # Skip if a file has already been traversed before as a reference image
                if file_name != other_file and other_file not in self.similar_images.keys():
                    score, _, _ = compare_frames_change_detection(image_hash, other_hash,
                                                                  self.min_contour_area, self.frame_change_thresh)
                    total_area = (image_hash.shape[0] * image_hash.shape[1])
                    # Normalize the dissimilarity score and compute the similarity score
                    similarity_score = 1 - (score / total_area)
                    if similarity_score > self.threshold:
                        self.similar_images[file_name].append([other_file, similarity_score])

    '''
    This method finds and removes similar-looking images from the specified folder. It first computes image hashes for
    each image in the folder and compares the hashes to find similar images. The detected similar images are saved to
    an output logger file. Then, the method proceeds to remove the similar images from the original folder, retaining
    only one representative image from each set of similar-looking images.
    '''
    def remove_similar_images(self):
        # Compute hashes for each image and store in a dictionary
        image_hashes = self._get_image_hashes()

        # Compare the hashes of images and find similar images
        self._find_similar_images(image_hashes)

        # Save the output (similar images) to a logger file
        self._write_to_logger()

        # Remove the similar-looking images from the folder
        for file_name, similar_files in self.similar_images.items():
            # Remove all similar files from the original folder
            for similar_file in similar_files:
                similar_path = os.path.join(self.folder_path, similar_file[0])
                dst_file_path = os.path.join(self.removed_images_path, similar_file[0])

                if os.path.exists(similar_path):
                    if not os.path.exists(dst_file_path):
                        # Move the similar file to the destination folder
                        shutil.copy(similar_path, self.removed_images_path)
                    # Remove the similar file from the original folder
                    os.remove(similar_path)
                    self.num_images_removed += 1
                else:
                    print(f"Warning: File '{similar_file[0]}' not found. Skipping deletion.")

        return self.similar_images, self.num_images_removed

    '''
    Selects a random image from the dataset, preprocesses it, and displays both the original and preprocessed images
    side by side. It is also saved in the current working directory as a .png image.
    '''
    def visualize_preprocess_image(self):
        # fFetch the list of all image files in the dataset
        image_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # randomly select an image from the list
        random_image = random.choice(image_files)
        random_image_path = os.path.join(self.folder_path, random_image)

        original_img = cv2.imread(random_image_path)
        preprocessed_img = preprocess_image_change_detection(original_img,
                                                            self.gaussian_blur_radius_list,
                                                            self.black_mask)
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(preprocessed_img, cmap='gray')
        plt.title("Preprocessed Image")
        plt.axis('off')

        plt.savefig(f'preprocessed_{random_image}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to dataset folder
    parser.add_argument('--folder_path', type=str, help='Absolute path to the folder containing dataset images')

    # Hyperparameters
    parser.add_argument('--threshold', type=float, default=0.85, help='Threshold for similarity score')
    parser.add_argument('--min_contour_area', type=int, default=500,
                        help='Minimum contour area for similarity comparison')
    parser.add_argument('--gaussian_blur_radius', type=str, nargs='+', default=["None"],
                        help='List of radii for Gaussian blur')
    parser.add_argument('--black_mask', type=int, nargs=4, default=[0, 15, 0, 0],
                        help='Black mask (in %) to apply on the image')
    parser.add_argument('--frame_change_thresh', type=int, default=25,
                        help='Threshold to convert grayscale images into binary')
    parser.add_argument('--resize_shape', type=int, nargs=2, default=[200, 200],
                        help='Size to reshape the images')
    args = parser.parse_args()

    folder_path = args.folder_path
    threshold = args.threshold
    min_contour_area = args.min_contour_area
    black_mask = tuple(args.black_mask)
    frame_change_thresh = args.frame_change_thresh
    resize_shape = tuple(args.resize_shape)
    # Convert "None" string to actual None value for gaussian_blur_radius_list
    gaussian_blur_radius_list = args.gaussian_blur_radius
    if len(gaussian_blur_radius_list) == 1 and gaussian_blur_radius_list[0] == "None":
        gaussian_blur_radius_list = None

    print("Folder Path:", folder_path)
    print("Threshold:", threshold)
    print("Minimum Contour Area:", min_contour_area)
    print("Gaussian Blur Radius List:", gaussian_blur_radius_list)
    print("Black Mask:", black_mask)
    print("Frame change threshold:", frame_change_thresh)
    print("Shape of resized image:", resize_shape)

    # Create an instance of the SimilarImagesRemover class
    remover = SimilarImagesRemover(folder_path, threshold, min_contour_area,
                                   gaussian_blur_radius_list, black_mask,
                                   frame_change_thresh, resize_shape)

    # Visualize a random image and its preprocessed version
    remover.visualize_preprocess_image()

    # Remove similar (duplicate or close to duplicate) images
    # The removed images are stored in a separate directory called "similar_images" for easy debugging
    similar_images, num_images_removed = remover.remove_similar_images()
    print(f"Removed {num_images_removed} redundant images.")