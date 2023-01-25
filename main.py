import numpy as np
import matplotlib.pyplot as plt

def calc_img(image):
    original_image = image.copy()
    # Convert to grayscale
    image = np.mean(image, axis=2)

    padding = [25, 1080-25-982, 10,1200-10-982] # [top, bottom, left, right] padding to only show the image

    # Cut out the image from the padding
    image = image[padding[0]:image.shape[0]-padding[1], padding[2]:image.shape[1]-padding[3]]
    original_image = original_image[padding[0]:original_image.shape[0]-padding[1], padding[2]:original_image.shape[1]-padding[3], :]

    # Normalize the image
    image = (image - np.min(image))/(np.max(image) - np.min(image))

    # Count each occurence of a pixel value to make a histogram
    histogram = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[int(image[i,j]*255)] += 1

    # Find the mean of the histogram
    mean = np.sum(histogram*np.arange(256))/np.sum(histogram)

    # Threshhold the mean
    image[image > mean/255.0] = 1
    image[image < mean/255.0] = 0

    # Count the number of pixels
    bright_pixels = np.sum(image)
    dark_pixels = image.shape[0]*image.shape[1] - bright_pixels

    # Plot the images
    plt.figure()
    plt.imshow(original_image)
    plt.title(f'AFM Aufnahme')

    # Add axis labels
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')

    # Rescale the axis label from 0 to 512
    plt.xticks([0, 982], [0, 512])
    plt.yticks([0, 982], [0, 512])

    plt.figure()
    plt.imshow(image)
    # Add title
    plt.title(f'Polystyrol (PS): {np.round(100*bright_pixels/(image.shape[0]*image.shape[1]),2)}%, Polyethylenpropylen (PEP): {np.round(100*dark_pixels/(image.shape[0]*image.shape[1]),2)}%')

    # Add colorbar
    plt.colorbar()

    # Add axis labels
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')

    # Rescale the axis label from 0 to 512
    plt.xticks([0, 982], [0, 512])
    plt.yticks([0, 982], [0, 512])

    return (dark_pixels, bright_pixels)


# Read images
d2, b2 = calc_img(plt.imread('230116_Kraton.0_00001_3.spm.jpg'))
d3, b3 = calc_img(plt.imread('230116_Kraton.0_00002_3.spm.jpg'))
d4, b4 = calc_img(plt.imread('230116_Kraton.0_00003_3.spm.jpg'))

# Image size = 982x982
pixelCout = 982*982

polystrol = np.array([b2, b3, b4])/(pixelCout)
polyethylenpropylen = np.array([d2, d3, d4])/(pixelCout)

# Calc errors
stdPolystrol = np.sqrt(np.sum((polystrol-np.mean(polystrol))**2)/3)
stdPolyethylenpropylen = np.sqrt(np.sum((polyethylenpropylen-np.mean(polyethylenpropylen))**2)/3)

# Round errors to 2 significant digits
stdPolystrol = np.round(stdPolystrol * 100, 2)
stdPolyethylenpropylen = np.round(stdPolyethylenpropylen * 100, 2)

# Round mean to 2 significant digits
meanPolystrol = np.round(np.mean(polystrol) * 100, 2)
meanPolyethylenpropylen = np.round(np.mean(polyethylenpropylen) * 100, 2)

# Print in table
print(f'Polystyrol (PS): {meanPolystrol} +- {stdPolystrol} %')
print(f'Polyethylenpropylen (PEP): {meanPolyethylenpropylen} +- {stdPolyethylenpropylen} %')

# Show plot
plt.show()
