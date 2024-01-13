# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Functions definitions
def blemishRemoval(action, x, y, flags, userdata):
    # Referencing global variables
    global r, src
    if action == cv2.EVENT_LBUTTONDOWN:
        # Store the blemish location in a variable to identify the region
        blemish_location = (x, y)
        print(f"Blemish Location at: {blemish_location}")
        # Call a function which calculates the best patch
        cropped_region = selectedRegion(x, y, r)  # This is a test function
        new_x, new_y = identifyBestPatch(x, y, r)  # We return the X and Y position to define the region to patch
        # We create a new patch with the new values:
        new_patch = src[new_y:(new_y + 2 * r), new_x:(new_x + 2 * r)]
        # Create a mask of ones using np.ones
        print(new_patch.shape, new_patch.dtype)
        mask = 255 * np.ones(new_patch.shape, new_patch.dtype)  # We should be careful with this
        src = cv2.seamlessClone(new_patch, src, mask, blemish_location, cv2.NORMAL_CLONE)
        cv2.imshow("Blemish removal App", src)
    elif action == cv2.EVENT_LBUTTONUP:
        cv2.imshow("Blemish removal App", src)


def selectedRegion(x, y, r):
    crop = src[y:y + 2 * r, x:x + 2 * r]
    # cv2.imshow("Cropped", crop) # Uncomment to see results
    return crop


def identifyBestPatch(x, y, r):
    # Define a patch dictionary
    patches = {}

    # Method 1: no difference encountered when looking for the best patch around (x, y)
    # This is due we don't move further outer the region it calculates all the key values based on x and y centers

    # # Define patch keys and their corresponding coordinates
    # patch_keys = []
    # for i in range(1,18):
    #     patch_keys.append("Key{}".format(i))
    #
    # # Loop over the patch keys and calculate gradient for each patch
    # for key in patch_keys:
    #     dx, dy = appendGradients(x, y, r)
    #     patches[key] = (x, y, dx, dy)

    # Method 2: we take the X and Y center of the original cropped image but now we move and create a new patch based
    # on the extreme point like (x+2*r, y+2*r) and pass it to the appendGradients as new_x and_y, and then calculate the
    # gradients, this works better because the values are not the same (as in the previous method), and comparisons
    # between them can be performed. Please refer to image attached in the same directory, filename: cartesian_explanation.jpeg

    # Define a list of 17 points that is the bounding box for the specified region
    key1_tuple = appendGradients(x + 2 * r, y, r)
    patches["Key1"] = (x + 2 * r, y, key1_tuple[0], key1_tuple[1])

    key2_tuple = appendGradients(x + 2 * r, y + r, r)
    patches["Key2"] = (x + 2 * r, y + r, key2_tuple[0], key2_tuple[1])

    key3_tuple = appendGradients(x + r, y + r, r)
    patches["Key3"] = (x + r, y + r, key3_tuple[0], key3_tuple[1])

    key4_tuple = appendGradients(x + 2 * r, y + 2 * r, r)
    patches["Key4"] = (x + 2 * r, y + 2 * r, key4_tuple[0], key4_tuple[1])

    key5_tuple = appendGradients(x + r, y + 2 * r, r)
    patches["Key5"] = (x + r, y + 2 * r, key5_tuple[0], key5_tuple[1])

    key6_tuple = appendGradients(x, y + 2 * r, r)
    patches["Key6"] = (x, y + 2 * r, key6_tuple[0], key6_tuple[1])

    key7_tuple = appendGradients(x - r, y + 2 * r, r)
    patches["Key7"] = (x - r, y + 2 * r, key7_tuple[0], key7_tuple[1])

    key8_tuple = appendGradients(x - 2 * r, y + 2 * r, r)
    patches["Key8"] = (x - 2 * r, y + 2 * r, key8_tuple[0], key8_tuple[1])

    key9_tuple = appendGradients(x - 2 * r, y + r, r)
    patches["Key9"] = (x - 2 * r, y + r, key9_tuple[0], key9_tuple[1])

    key10_tuple = appendGradients(x - 2 * r, y, r)
    patches["Key10"] = (x - 2 * r, y, key10_tuple[0], key10_tuple[1])

    key11_tuple = appendGradients(x - 2 * r, y - r, r)
    patches["Key11"] = (x - 2 * r, y - r, key11_tuple[0], key11_tuple[1])

    key12_tuple = appendGradients(x - 2 * r, y - 2 * r, r)
    patches["Key12"] = (x - 2 * r, y - 2 * r, key12_tuple[0], key12_tuple[1])

    key13_tuple = appendGradients(x - r, y - 2 * r, r)
    patches["Key13"] = (x - r, y - 2 * r, key13_tuple[0], key13_tuple[1])

    key14_tuple = appendGradients(x, y - 2 * r, r)
    patches["Key14"] = (x, y - 2 * r, key14_tuple[0], key14_tuple[1])

    key15_tuple = appendGradients(x + r, y - 2 * r, r)
    patches["Key15"] = (x + r, y - 2 * r, key15_tuple[0], key15_tuple[1])

    key16_tuple = appendGradients(x + 2 * r, y - 2 * r, r)
    patches["Key16"] = (x + 2 * r, y - 2 * r, key16_tuple[0], key16_tuple[1])

    key17_tuple = appendGradients(x + 2 * r, y - r, r)
    patches["Key17"] = (x + 2 * r, y - r, key17_tuple[0], key17_tuple[1])

    # Method 3: Is the same as Method 2 but using fewer lines of code to iterate over the key values (more efficient)
    # for row in range(-2, 3):
    #     for col in range(-2, 3):
    #         key = f"Key{row * 5 + col + 18}"  # Calculate the key based on row and column
    #         dx, dy = appendGradients(x + col * r * 2, y + row * r * 2, r)  # Calculate gradients
    #         patches[key] = (x + col * r * 2, y + row * r * 2, dx, dy)  # Store data in patches
    # print(patches)

    # Now we define two new dicts where they store the gradients in X and y
    find_low_x = {}
    find_low_y = {}
    for key, (x, y, gx, gy) in patches.items():
        find_low_x[key] = gx
        find_low_y[key] = gy

    # Now we find the lowest value in both dicts
    x_min_key = min(find_low_x.keys(), key=find_low_x.get)
    y_min_key = min(find_low_y.keys(), key=find_low_y.get)

    if x_min_key == y_min_key:
        print("Sobel analysis successful...")
        return patches[x_min_key][0], patches[x_min_key][1]
    else:
        # Because we analyzed before the where is the minimum we don't need to go through all the patches again,
        # so we take x_min_key and y_min_key along with the patches, and we perform a frequency analysis over the
        # cropped regions related to those min key values and determine which of those has the lowest frequency
        print("Sobel analysis was not successful. Performing FFT analysis instead...")
        # This function will return the best key to select the best patch based on visual frequency analysis,
        # the user can select between the two patches with the lowest frequencies and analyze the frequency domain of
        # each one. PLEASE PAY ATTENTION TO THE PROMPT, an image with the name of magnitude_images_with_colorbars.png
        # will be saved in the same directory to perform the analysis
        key = fftAnalysis(x_min_key, y_min_key, patches, r)
        print(patches[key][0], patches[key][1])
        return patches[key][0], patches[key][1]


def appendGradients(x, y, r):
    # crop_image = src[y-r:y+r, x-r:x+r] # Center over original x and y positions
    crop_image = src[y:(y + 2 * r), x:(x + 2 * r)]  # Center over the X and Y passed in identifyBestPatch
    gradient_x, gradient_y = sobelFilter(crop_image)
    return gradient_x, gradient_y


def sobelFilter(crop_image):
    # Calculate Sobel gradient for X-direction
    sobelx64f = cv2.Sobel(crop_image, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx64f = np.absolute(sobelx64f)
    sobelx_8u = np.uint8(abs_sobelx64f)
    gradient_x = np.mean(sobelx_8u)

    # Calculate Sobel gradient for Y-direction
    sobely64f = cv2.Sobel(crop_image, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely64f = np.absolute(sobely64f)
    sobely_8u = np.uint8(abs_sobely64f)
    gradient_y = np.mean(sobely_8u)

    return gradient_x, gradient_y


def fftAnalysis(x_min_key, y_min_key, patches, r):
    # Crop the image to the related patches
    cropped_image_x_min_key = src[patches[x_min_key][1]:patches[x_min_key][1] + 2 * r,
                              patches[x_min_key][0]:patches[x_min_key][0] + 2 * r]
    cropped_image_y_min_key = src[patches[y_min_key][1]:patches[y_min_key][1] + 2 * r,
                              patches[y_min_key][0]:patches[y_min_key][0] + 2 * r]

    # Transform croped regions to gray scale
    cropped_image_x_min_key_grayscale = cv2.cvtColor(cropped_image_x_min_key, cv2.COLOR_BGR2GRAY)
    cropped_image_y_min_key_grayscale = cv2.cvtColor(cropped_image_y_min_key, cv2.COLOR_BGR2GRAY)

    # Expand the image to the optimal size
    # For x_min_key:
    rows_x_min_key, cols_x_min_key = cropped_image_x_min_key_grayscale.shape
    m_image_x_min_key = cv2.getOptimalDFTSize(rows_x_min_key)
    n_image_x_min_key = cv2.getOptimalDFTSize(cols_x_min_key)
    padded_x_min_key = cv2.copyMakeBorder(cropped_image_x_min_key_grayscale, 0, m_image_x_min_key - rows_x_min_key, 0,
                                          n_image_x_min_key - cols_x_min_key, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # For y_min_key:
    rows_y_min_key, cols_y_min_key = cropped_image_y_min_key_grayscale.shape
    m_image_y_min_key = cv2.getOptimalDFTSize(rows_y_min_key)
    n_image_y_min_key = cv2.getOptimalDFTSize(cols_y_min_key)
    padded_y_min_key = cv2.copyMakeBorder(cropped_image_y_min_key_grayscale, 0, m_image_y_min_key - rows_y_min_key, 0,
                                          n_image_y_min_key - cols_y_min_key, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Make place for both the complex and the real values
    # For x_min_key:
    planes_x_min_key = [np.float32(padded_x_min_key), np.zeros(padded_x_min_key.shape, np.float32)]
    complexI_x_min_key = cv2.merge(planes_x_min_key)
    # For y_min_key:
    planes_y_min_key = [np.float32(padded_y_min_key), np.zeros(padded_y_min_key.shape, np.float32)]
    complexI_y_min_key = cv2.merge(planes_y_min_key)  # Add to the expanded another plane with zeros

    # Make the Discrete Fourier Transform
    # For x_min_key:
    dft_x_min_key = cv2.dft(complexI_x_min_key)  # this way the result may fit in the source matrix
    # For y_min_key:
    dft_y_min_key = cv2.dft(complexI_y_min_key)  # this way the result may fit in the source matrix

    # Transform the real and complex values to magnitude
    # For x_min_key:
    planes_x_min_key = cv2.split(dft_x_min_key)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    mag_x_min_key = cv2.magnitude(planes_x_min_key[0], planes_x_min_key[1])  # planes[0] = magnitude
    # For y_min_key:
    planes_y_min_key = cv2.split(dft_y_min_key)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    mag_y_min_key = cv2.magnitude(planes_y_min_key[0], planes_y_min_key[1])  # planes[0] = magnitude

    # Switch to a logarithmic scale
    # For x_min_key:
    mat_of_ones_x_min_key = np.ones(mag_x_min_key.shape, dtype=mag_x_min_key.dtype)
    mag_x_min_key = cv2.add(mat_of_ones_x_min_key, mag_x_min_key)  # switch to logarithmic scale
    mag_x_min_key = cv2.log(mag_x_min_key)
    # For y_min_key:
    mat_of_ones_y_min_key = np.ones(mag_y_min_key.shape, dtype=mag_y_min_key.dtype)
    mag_y_min_key = cv2.add(mat_of_ones_y_min_key, mag_y_min_key)  # switch to logarithmic scale
    mag_y_min_key = cv2.log(mag_y_min_key)

    # Crop and rearrange
    # For x_min_key:
    mag_rows_x_min_key, mag_cols_x_min_key = mag_x_min_key.shape
    # crop the spectrum, if it has an odd number of rows or columns
    mag_x_min_key = mag_x_min_key[0:(mag_rows_x_min_key & -2), 0:(mag_cols_x_min_key & -2)]
    cx_x_min_key = int(mag_rows_x_min_key / 2)
    cy_x_min_key = int(mag_cols_x_min_key / 2)
    q0_x_min_key = mag_x_min_key[0:cx_x_min_key, 0:cy_x_min_key]  # Top-Left - Create a ROI per quadrant
    q1_x_min_key = mag_x_min_key[cx_x_min_key:cx_x_min_key + cx_x_min_key, 0:cy_x_min_key]  # Top-Right
    q2_x_min_key = mag_x_min_key[0:cx_x_min_key, cy_x_min_key:cy_x_min_key + cy_x_min_key]  # Bottom-Left
    q3_x_min_key = mag_x_min_key[cx_x_min_key:cx_x_min_key + cx_x_min_key, cy_x_min_key:cy_x_min_key + cy_x_min_key]  # Bottom-Right
    tmp_x_min_key = np.copy(q0_x_min_key)  # swap quadrants (Top-Left with Bottom-Right)
    mag_x_min_key[0:cx_x_min_key, 0:cy_x_min_key] = q3_x_min_key
    mag_x_min_key[cx_x_min_key:cx_x_min_key + cx_x_min_key, cy_x_min_key:cy_x_min_key + cy_x_min_key] = tmp_x_min_key
    tmp_x_min_key = np.copy(q1_x_min_key)  # swap quadrant (Top-Right with Bottom-Left)
    mag_x_min_key[cx_x_min_key:cx_x_min_key + cx_x_min_key, 0:cy_x_min_key] = q2_x_min_key
    mag_x_min_key[0:cx_x_min_key, cy_x_min_key:cy_x_min_key + cy_x_min_key] = tmp_x_min_key

    # For y_min_key:
    mag_rows_y_min_key, mag_cols_y_min_key = mag_y_min_key.shape
    # crop the spectrum, if it has an odd number of rows or columns
    mag_y_min_key = mag_y_min_key[0:(mag_rows_y_min_key & -2), 0:(mag_cols_y_min_key & -2)]
    cx_y_min_key = int(mag_rows_y_min_key / 2)
    cy_y_min_key = int(mag_cols_y_min_key / 2)
    q0_y_min_key = mag_y_min_key[0:cx_y_min_key, 0:cy_y_min_key]  # Top-Left - Create a ROI per quadrant
    q1_y_min_key = mag_y_min_key[cx_y_min_key:cx_y_min_key + cx_y_min_key, 0:cy_y_min_key]  # Top-Right
    q2_y_min_key = mag_y_min_key[0:cx_y_min_key, cy_y_min_key:cy_y_min_key + cy_y_min_key]  # Bottom-Left
    q3_y_min_key = mag_y_min_key[cx_y_min_key:cx_y_min_key + cx_y_min_key, cy_y_min_key:cy_y_min_key + cy_y_min_key]  # Bottom-Right
    tmp_y_min_key = np.copy(q0_y_min_key)  # swap quadrants (Top-Left with Bottom-Right)
    mag_y_min_key[0:cx_y_min_key, 0:cy_y_min_key] = q3_y_min_key
    mag_y_min_key[cx_y_min_key:cx_y_min_key + cx_y_min_key, cy_y_min_key:cy_y_min_key + cy_y_min_key] = tmp_y_min_key
    tmp_y_min_key = np.copy(q1_y_min_key)  # swap quadrant (Top-Right with Bottom-Left)
    mag_y_min_key[cx_y_min_key:cx_y_min_key + cx_y_min_key, 0:cy_y_min_key] = q2_y_min_key
    mag_y_min_key[0:cx_y_min_key, cy_y_min_key:cy_y_min_key + cy_y_min_key] = tmp_y_min_key

    # Normalize
    # For x_min_key:
    mag_x_min_key = cv2.normalize(mag_x_min_key, 0, 1, cv2.NORM_MINMAX)
    mag_x_min_key = np.uint8(mag_x_min_key * 255)
    # For y_min_key:
    mag_y_min_key = cv2.normalize(mag_y_min_key, 0, 1, cv2.NORM_MINMAX)
    mag_y_min_key = np.uint8(mag_y_min_key * 255)

    # Display normalized magnitude images with color bars
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display magnitude image with color bar for X
    im1 = axs[0].imshow(mag_x_min_key, cmap='jet')
    axs[0].set_title(f"Normalized Magnitude of {x_min_key}")
    axs[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axs[0])
    cbar1.set_label('Magnitude')

    # Display magnitude image with color bar for Y
    im2 = axs[1].imshow(mag_y_min_key, cmap='jet')
    axs[1].set_title(f"Normalized Magnitude {y_min_key}")
    axs[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axs[1])
    cbar2.set_label('Magnitude')

    plt.tight_layout()
    # Save the figure to an image file (e.g., PNG)
    plt.savefig("magnitude_images_with_colorbars.png", dpi=300)  # Set dpi as needed

    while True:
        user_input = input(f"Please select 1 to apply patch {x_min_key}\n"
                           f"or select 2 to apply patch {y_min_key}: ")

        if user_input == '1':
            return x_min_key
        elif user_input == '2':
            return y_min_key
        else:
            print("Invalid input. Please select 1 or 2.")


# Execute the program Main
if __name__ == "__main__":
    # Read the image
    src = cv2.imread(r"..\Blemish removal\blemish.png", cv2.IMREAD_UNCHANGED)

    # Lists to store the points
    r = 15

    # Make a dummy image, it will be useful to clear the drawing
    dummy = src.copy()
    cv2.namedWindow("Blemish removal App")

    # Call the blemish function when mouse event occurs
    cv2.setMouseCallback("Blemish removal App", blemishRemoval)
    k = 0

    # Show the image until esc is pressed
    while k != 27:
        cv2.imshow("Blemish removal App", src)
        k = cv2.waitKey(20) & 0xFF
        # Clean the drawing
        if k == 99:
            src = dummy.copy()

    cv2.destroyAllWindows()