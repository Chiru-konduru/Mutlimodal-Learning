## Multi-Model Learning: Image Captioning with Conventional and CLIP + GPT-2 Models

**Repository Overview:**

This repository hosts code, datasets, and models developed for our image captioning project. The project's objective is to design, train, and evaluate deep learning models capable of generating accurate and contextually relevant text descriptions for input images.

# Models Explored:

# Conventional Vision-Language Model (VLM): This model uses VGG16 (a convolutional neural network) to extract image features and an LSTM network to generate word sequences that describe the images.
# CLIPCap Model: Our primary model that integrates the CLIP (Contrastive Language-Image Pretraining) model with GPT-2 for image captioning. CLIP learns to relate text and images, making it well-suited for captioning tasks. The GPT-2 integration ensures coherent and contextually appropriate captions based on image features detected by CLIP.
Detailed Project Summary:
Our project aims to develop and assess a deep learning model capable of generating meaningful text descriptions for given images. We explore both conventional architectures and an innovative design that integrates CLIP with GPT-2.

Conventional VLMs are typically trained on datasets that pair text and images. They can perform various tasks like image captioning, visual question answering, and visual dialogue. In our project, the baseline model combines the VGG16 CNN with LSTM networks: VGG16 extracts image features, and the LSTM produces a sequence of words to depict the image.

Our primary model, CLIPCap, incorporates the CLIP model with GPT-2. CLIP's pretraining on associating images and text makes it potent for image captioning tasks. Training CLIPCap on both a text-image paired dataset and a dataset containing human-generated image captions allows the model to understand relationships between text and images, as well as mimic human-like image descriptions.

# Training Details:

The baseline model (VGG16+LSTMs) is trained on the Flickr8k dataset, familiarizing the model with the connections between image content and natural language descriptions.
For the main CLIPCap model, we fine-tune the pre-trained CLIP model using the same Flickr8k dataset to align with our image captioning objective.
