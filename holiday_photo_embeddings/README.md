## Getting started
To run the code, you need to have a folder containing images you want to generate embeddings from. Replace hardcoded paths
to this folder and destination folder in util.py and run it to crop, resize and save the modified images in specified folder.
Replace the hardcoded path to the destination folder in img_mbed.py and run it to generate embeddings.

With the embeddings generated, you can play with three different algorithms implemented in analyzer.py: clustering, nearest
neighbor search and finding best match for difference of two images.
