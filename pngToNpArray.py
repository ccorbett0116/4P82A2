import numpy as np
from PIL import Image
##reading in
#img = Image.open('test128.png')
#data = np.array(img.convert('RGB'))
##showing
#new_image = Image.fromarray(data, 'RGB')
#new_image.show()
##saving
##new_image.save('output_image.png')
# Function to read image and return as numpy array
def readImage(imagePath):
    img = Image.open(imagePath)
    data = np.array(img.convert('RGB'))
    return data
# Function to save image from numpy array
def saveImage(image, outputPath):
    new_image = Image.fromarray(image, 'RGB')
    new_image.save(outputPath)

