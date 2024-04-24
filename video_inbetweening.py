# Import the required libraries
!pip install opendatasets  # Install the opendatasets library for downloading datasets
import opendatasets as od  # Import the opendatasets library for dataset operations
import pandas  # Import the pandas library for data manipulation
import tensorflow as tf  # Import TensorFlow for deep learning tasks

import matplotlib.pyplot as plt  # Import matplotlib for data visualization
import numpy as np  # Import numpy for numerical operations
import seaborn as sns  # Import seaborn for statistical data visualization
import tensorflow_hub as hub  # Import tensorflow_hub for TensorFlow Hub modules
import tensorflow_datasets as tfds  # Import tensorflow_datasets for accessing TensorFlow datasets

from tensorflow_datasets.core import SplitGenerator  # Import SplitGenerator for generating dataset splits
from tensorflow_datasets.video.bair_robot_pushing import BairRobotPushingSmall  # Import BairRobotPushingSmall dataset from TensorFlow Datasets

import tempfile  # Import tempfile for creating temporary directories
import pathlib  # Import pathlib for handling file paths

# Define the directory path for storing the test data of the BairRobotPushingSmall dataset
TEST_DIR = pathlib.Path(tempfile.mkdtemp()) / "bair_robot_pushing_small/softmotion30_44k/test/"

# Downloading the test split to $TEST_DIR
!mkdir -p $TEST_DIR
!wget -nv https://storage.googleapis.com/download.tensorflow.org/data/bair_test_traj_0_to_255.tfrecords -O $TEST_DIR/traj_0_to_255.tfrecords

# Create an instance of the BairRobotPushingSmall dataset builder
builder = BairRobotPushingSmall()

# Define a generator for the test split of the dataset, specifying the directory containing the test data
test_generator = SplitGenerator(name='test', gen_kwargs={"filedir": str(TEST_DIR)})

# Patch the dataset builder to use the test generator instead of the default split generators
builder._split_generators = lambda _: [test_generator]

# Download and prepare the dataset using the specified split generators
builder.download_and_prepare()

batch_size = 16  # Define the batch size for processing the dataset in batches

# Load the test split dataset using the BairRobotPushingSmall builder
ds = builder.as_dataset(split="test")

# Batch the test dataset into batches of size defined by batch_size
test_videos = ds.batch(batch_size)

# Extract the first batch of videos from the test dataset for processing
first_batch = next(iter(test_videos))

# Extract the input frames (start and end frames) from the first batch
input_frames = first_batch['image_aux1'][:, ::15]

# Convert the input frames to float32 data type using TensorFlow
input_frames = tf.cast(input_frames, tf.float32)

# Print the shape of the input frames tensor, indicating the dimensions of the videos
print('Test videos shape [batch_size, start/end frame, height, width, num_channels]: ', input_frames.shape)

# Set the style of seaborn plots to 'white' for better visualization
sns.set_style('white')

# Create a new matplotlib figure with a specific size based on the batch size
plt.figure(figsize=(4, 2*batch_size))

# Loop through the batch of input frames and plot the first and last frames of each video
for i in range(batch_size)[:4]:
    # Plot the first frame of the current video
    plt.subplot(batch_size, 2, 1 + 2*i)
    plt.imshow(input_frames[i, 0] / 255.0)  # Normalize and display the image
    plt.title('Video {}: First frame'.format(i))  # Set the title indicating the frame type
    plt.axis('off')  # Turn off axis labels for better visualization

    # Plot the last frame of the current video
    plt.subplot(batch_size, 2, 2 + 2*i)
    plt.imshow(input_frames[i, 1] / 255.0)  # Normalize and display the image
    plt.title('Video {}: Last frame'.format(i))  # Set the title indicating the frame type
    plt.axis('off')  # Turn off axis labels for better visualization

# Generate intermediate frames using the loaded model and input frames
filled_frames = module(input_frames)['default'] / 255.0

# Concatenate the start frames, filled frames, and end frames to create the complete generated videos
generated_videos = np.concatenate([input_frames[:, :1] / 255.0, filled_frames, input_frames[:, 1:] / 255.0], axis=1)

# Visualize sequences of generated video frames
for video_id in range(4):
    # Create a new matplotlib figure for each video sequence
    fig = plt.figure(figsize=(10 * 2, 2))
    for frame_id in range(1, 16):
        # Add an axis to the figure for each frame in the sequence
        ax = fig.add_axes([frame_id * 1 / 16., 0, (frame_id + 1) * 1 / 16., 1], xmargin=0, ymargin=0)
        # Display the corresponding generated frame
        ax.imshow(generated_videos[video_id, frame_id])
        ax.axis('off')  # Turn off axis labels for better visualization

