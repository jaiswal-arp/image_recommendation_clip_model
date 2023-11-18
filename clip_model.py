import os
import gdown
import zipfile
from pathlib import Path
import shutil
import math
import numpy as np
import pandas as pd
from PIL import Image
import clip
import torch
import csv
import math
import pinecone
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 'truetype'})
from mpl_toolkits.axes_grid1 import ImageGrid


# Set the shared Google Drive link to the zip file containing your dataset
google_drive_link = 'https://drive.google.com/uc?id=1-71sckG0w8sFL5Zf5hngVd4WfnH7vviT'

# Set the desired root directory in Colab
colab_root = '/content/test'

# Define the path where you want to store the dataset
dataset_path = os.path.join(colab_root, 'img')

# Create the dataset directory if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

# Download the zip file using gdown
zip_file_path = os.path.join(dataset_path, 'img.zip')
gdown.download(google_drive_link, output=zip_file_path, quiet=False)

# Unzip the downloaded file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# Set the path to the images
images_path = Path('/content/test/img/img/MEN/Denim/id_00000080')

# List all JPGs in the folder
images_files = list(images_path.glob('*.jpg'))

# Print some statistics
print(f'Images found: {len(images_files)}')

# Display images
rows, columns = (4, 4)
fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(rows, columns),
                 axes_pad=0.1,
                 )

for ax, im in zip(grid, images_files[:rows*columns]):
    ax.imshow(Image.open(im))
    ax.axis('off')
plt.show()


# Set the root path 
root_path = Path('/content/test/img/img/MEN')

# List all subfolders 
category_folders = [folder for folder in root_path.iterdir() if folder.is_dir()]
print(category_folders)


# Initialize CLIP model and preprocessing functions
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the root directory of your image dataset
root_dir = Path("/content/test/img/img/MEN")

# Collect all image files recursively from the specified directory
image_files = list(root_dir.glob("**/*.jpg"))  # Adjust the file extension as needed

# Path where the feature vectors will be stored
features_path = Path(".") / "features"
if os.path.exists(features_path):
    shutil.rmtree(features_path)
os.makedirs(features_path)

# Initialize Pinecone client
pinecone = api_key="4628f9ab-aef0-4667-8568-c6a6b29567e8"
index_name = "imageindex"

# Function that computes the feature vectors for a batch of images
def compute_clip_features(images_batch):
    # Load all the images from the files
    images = [Image.open(image_file) for image_file in images_batch]

    # Preprocess all images
    images_preprocessed = torch.stack([preprocess(image) for image in images]).to(device)

    with torch.no_grad():
        # Encode the images batch to compute the feature vectors and normalize them
        images_features = model.encode_image(images_preprocessed)
        images_features /= images_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return images_features.cpu().numpy()

# Define the batch size
batch_size = 16

# Compute how many batches are needed
batches = math.ceil(len(image_files) / batch_size)

# Store feature vectors in a list during initial processing
feature_vectors_list = []

# Process each batch
for i in range(batches):
    print(f"Processing batch {i + 1}/{batches}")

    batch_ids_path = features_path / f"{i:010d}.csv"
    batch_features_path = features_path / f"{i:010d}.npy"

    # Only do the processing if the batch wasn't processed yet
    if not batch_features_path.exists():
        try:
            # Select the images for the current batch
            batch_files = image_files[i * batch_size : (i + 1) * batch_size]

            # Compute the features and save to a numpy file
            batch_features = compute_clip_features(batch_files)
            np.save(batch_features_path, batch_features)

            # Save the image IDs to a CSV file
            image_ids = [image_file.relative_to(root_dir).with_suffix("").as_posix() for image_file in batch_files]
            image_ids_data = pd.DataFrame(image_ids, columns=['image_id'])
            image_ids_data.to_csv(batch_ids_path, index=False)

            feature_vectors_list.append((image_ids, batch_features.tolist()))
            print(batch_features.tolist())
            
        except Exception as e:
            # Catch problems with the processing to make the process more robust
            print(f'Problem with batch {i} {e}')

# Optionally, load all numpy files
features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

# Initialize Pinecone client
pinecone.init(api_key="4628f9ab-aef0-4667-8568-c6a6b29567e8", environment="gcp-starter")
index = pinecone.Index("imageindex")


values=[]
for x in feature_vectors_list:
  for idx in range(len(x[0])):
      try:
         obj={
            "id":x[0][idx],
            "values":x[1][idx]
            }
         values.append(obj)
      except Exception as e:
          # Handle upsert errors
          print(f'Problem with upsert: {e}')


#inserting embeddings in batches
def split_into_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
batches = split_into_batches(values, 100)

#Inserting values in pinecone index
for batch in batches:
  index.upsert(vectors=batch)


def encode_search_query(search_query):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    # Retrieve the feature vector
    return text_encoded

def find_best_matches(text_features, image_features, image_ids, results_count=3):
  # Compute the similarity between the search query and each image using the Cosine similarity
  similarities = (image_features @ text_features.T).squeeze(1)
   # Sort the images by their similarity score
  best_image_idx = (-similarities).argsort()

  # Return the image IDs of the best matches
  return [image_ids[i] for i in best_image_idx[:results_count]]

def search(search_query, image_features, image_ids, results_count=3):
  # Encode the search query
  text_features = encode_search_query(search_query)

  # Find the best matches
  return find_best_matches(text_features, image_features, image_ids, results_count)

#
# Search for images and visualize the results
#
search_queries = ['Leather Biker Jackets',
                  'Graphic Tees Collection',
                  'Athletic Tanks for Gym',
                  'Pullover Hooded Sweatshirts',
                  'Ripped Denim Jeans',
                  'A Blue Pant', 
                  'A black Denim',
                  'A rainbow sweater'
                  ]
n_results_per_query = 3
results_dict= {} 

fig, ax = plt.subplots(len(search_queries), n_results_per_query + 1, figsize=(15, 10))    
for i, search_query in enumerate(search_queries):
    result_image_ids = search(search_query, image_features, image_ids, n_results_per_query)
    results_dict[search_query] = result_image_ids
    
    ax[i, 0].text(0.0, 0.5, search_query)
    ax[i, 0].axis('off')
    for j, image_id in enumerate(result_image_ids):
        img_path=f'{root_path}/{image_id}.jpg'  
        print(img_path)
        image = Image.open(f'{root_path}/{image_id}.jpg')
        ax[i, j+1].imshow(image)
        ax[i, j+1].axis('off')




# Function to save results to a CSV file
def save_results_to_csv(results, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Query', 'Image_IDs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the results for each query
        for query, image_ids in results.items():
            writer.writerow({'Query': query, 'Image_IDs': ', '.join(image_ids)})


#Save results to CSV
csv_filename = 'search_results.csv'
save_results_to_csv(results_dict, csv_filename)

# Display the CSV filename
print(f'Results saved to {csv_filename}')
