import os
from typing import Any, List

import cv2
import faiss
import numpy as np


def load_images_from_folder(folder_path: str) -> List[Any]:
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))

        if img is not None:
            images.append(img)
    return images


def extract_sift_features(images: List):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


def create_visual_dictionary(descriptors: np.ndarray, num_clusters: int) -> Any:
    d = descriptors.shape[1]  # Dimension of each vector
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=300)
    kmeans.train(descriptors.astype(np.float32))
    return kmeans


def generate_feature_histograms(
    descriptors: np.ndarray, visual_dictionary: Any
) -> List[Any]:
    num_clusters = visual_dictionary.k
    histograms = []

    for desc in descriptors:
        histogram = np.zeros(num_clusters)
        _, labels = visual_dictionary.index.search(desc.astype(np.float32), 1)
        for label in labels.flatten():
            histogram[label] += 1
        histograms.append(histogram)

    return histograms


# def compare_histograms(query_histogram, list_of_histograms: List[Any]) -> int:
#     # Calculate Euclidean distances
#     distances = [np.linalg.norm(query_histogram - hist) for hist in list_of_histograms]

#     # Find the index of the most similar histogram
#     most_similar_index = np.argmin(distances)

#     return most_similar_index


def process_image_and_find_best_match(
    new_image: np.ndarray, list_of_histograms: List[Any], kmeans: Any
) -> np.ndarray:
    # Step 1: Extract features from the new image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(new_image, None)

    # Ensure descriptors are in the correct format (np.float32)
    descriptors = descriptors.astype(np.float32)

    # Step 2: Generate the feature histogram for the new image
    num_clusters = kmeans.k
    histogram = np.zeros(num_clusters)

    # Use FAISS to find nearest clusters
    _, labels = kmeans.index.search(descriptors, 1)
    for label in labels.flatten():
        histogram[label] += 1

    # Step 3: Compare the histogram to the list of histograms
    distances = [np.linalg.norm(histogram - hist) for hist in list_of_histograms]

    # Find the indices of the 3 best candidates
    best_candidates_indices = np.argsort(distances)[:3]

    return np.array(best_candidates_indices)


# For testing only
def main():
    folder_path = "data/textures"
    images = load_images_from_folder(folder_path)
    keypoints, descriptors = extract_sift_features(images)
    visual_dictionary = create_visual_dictionary(
        np.vstack(descriptors), num_clusters=100
    )
    histograms = generate_feature_histograms(descriptors, visual_dictionary)

    best_indexes = (
        images[155], histograms, visual_dictionary
    )

    for index in best_indexes:
        cv2.imshow(f"{index}th best guess", images[index])
        cv2.waitKey(0)


if __name__ == "__main__":
    main()