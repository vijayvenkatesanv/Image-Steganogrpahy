import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2
import json

def read_and_convert_image(image_path):
    image = cv2.imread(image_path)  #to read the image file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #converting BGR to RGB
    return image

def dwt_coefficients(image, wavelet):
    cA, (cH, cV, cD) = pywt.dwt2(image, wavelet)
    return cA, cH, cV, cD

def show_coefficients_subplot(coefficients, subplot_base, dwt_labels):
    for i, a in enumerate(coefficients):
        subplot_number = subplot_base + i
        plt.subplot(4, 4, subplot_number)
        plt.title(dwt_labels[i])
        plt.axis('off')
        plt.imshow(a, interpolation="nearest", cmap=plt.cm.gray)

def svd_decomposition(matrix):
    P, D, Q = np.linalg.svd(matrix, full_matrices=False)
    return P, D, Q
    
# Function to validate the password for decoding
def validate_password(patient_choice, entered_password, patients_data):
    if patient_choice in patients_data:
        patient_info = patients_data[patient_choice]
        return entered_password == patient_info['Password']
    return False

wavelet = "db20"  # indicates which waveform (wavelet) was used for the wavelet transform

def load_patient_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# Load patient data from JSON
patients_data = load_patient_data('patient_details.json')

# Ask for patient choice
while True:
    patient_choice = input("\nWhich patient's medical data do you wish to see? (Proakis, Simon, Shreya): ")

    if patient_choice in patients_data:
        data_image_path = patients_data[patient_choice]['Image_to_Hide']
        cover_image_path = patients_data[patient_choice]['Image_to_Send']

        data_image = read_and_convert_image(data_image_path)
        cover_image = read_and_convert_image(cover_image_path)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Cover Image")
        plt.imshow(cover_image, aspect="equal")

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Image to Hide")
        plt.imshow(data_image, aspect="equal")

        plt.show()

        # --------------------------------------------
        # Encoding Process
        # Process of hiding an image within another image
        # ---------------------------------------------

        #Separating channels (colors) for cover and hidden images
        cover_image_r = cover_image[:, :, 0] #extracting the red channel for cover image
        cover_image_g = cover_image[:, :, 1] #extracting the green channel for cover image
        cover_image_b = cover_image[:, :, 2] #extracting the blue channel for cover image

        data_image_r = data_image[:, :, 0] #extracting the red channel for data image
        data_image_g = data_image[:, :, 1] #extracting the green channel for data image
        data_image_b = data_image[:, :, 2] #extracting the blue channel for data image

        # Apply the wavelet transform to the cover and hidden images
        cAr, cHr, cVr, cDr = dwt_coefficients(cover_image_r, wavelet)  #gets the approximation, horizontal, vertical and diagonal coefficients for red channel of cover image
        cAg, cHg, cVg, cDg = dwt_coefficients(cover_image_g, wavelet)  #gets the approximation, horizontal, vertical and diagonal coefficients for green channel of cover image
        cAb, cHb, cVb, cDb = dwt_coefficients(cover_image_b, wavelet)  #gets the approximation, horizontal, vertical and diagonal coefficients for blue channel of cover image

        cAr1, cHr1, cVr1, cDr1 = dwt_coefficients(data_image_r, wavelet)  #gets the approximation, horizontal, vertical and diagonal coefficients for red channel of data image
        cAg1, cHg1, cVg1, cDg1 = dwt_coefficients(data_image_g, wavelet)  #gets the approximation, horizontal, vertical and diagonal coefficients for green channel of data image
        cAb1, cHb1, cVb1, cDb1 = dwt_coefficients(data_image_b, wavelet)  #gets the approximation, horizontal, vertical and diagonal coefficients for blue channel of data image

        # Plot all layers resulted from DWT
        dwt_labels = ["Approximation", "Horizontal Detail", "Vertical Detail", "Diagonal Detail"]

        plt.figure(figsize=(10, 10))

        #plotting the coefficient details
        show_coefficients_subplot([cAr, cHr, cVr, cDr], 5, dwt_labels)
        show_coefficients_subplot([cAr1, cHr1, cVr1, cDr1], 9, dwt_labels)

        # Perform Singular Value Decomposition (SVD) on the cover and hidden images
        
        #P: Left singular value matrix vectors that form an orthonormal basis for the column space of the original matrix 
        #Each column vector in P represents a direction in the input space.
        #D: Singular values 1-D array that describes the "importance" or "strength" of each corresponding singular vector in P and Q
        #Q: Right singular value matrix vectors that form an orthonormal basis for the row space of the original matrix
        #Each row vector in Q corresponds to a direction in the output space.
        
        Pr, Dr, Qr = svd_decomposition(cAr)
        Pg, Dg, Qg = svd_decomposition(cAg)
        Pb, Db, Qb = svd_decomposition(cAb)

        P1r, D1r, Q1r = svd_decomposition(cAr1)
        P1g, D1g, Q1g = svd_decomposition(cAg1)
        P1b, D1b, Q1b = svd_decomposition(cAb1)

        # Embed the hidden information into the 'D' parameters of the cover image
        # The singular values are embedded by using a scaling factor
        # More the scaling factor: better the embedding but image quality will be lost
        # So an ideal value of 0.1 is chosen here
        Embed_Dr = Dr + (0.1 * D1r)
        Embed_Dg = Dg + (0.1 * D1g)
        Embed_Db = Db + (0.1 * D1b)
        
        # Reconstruct the coefficient matrix from the embedded SVD parameters
        # by scalar matrix multiplcation
        imgr = np.dot(Pr * Embed_Dr, Qr);
        imgg = np.dot(Pg * Embed_Dg, Qg)
        imgb = np.dot(Pb * Embed_Db, Qb)

        # Converting the image to int 0-255 range from float resulted in SVD computation
        a = imgr.astype(int)  # red matrix
        b = imgg.astype(int)  # green matrix
        c = imgb.astype(int)  # blue matrix

        # Concatenate the three reconstructed RGB channels into a single matrix
        img = cv2.merge((a, b, c))

        # Extract the horizontal, vertical, and diagonal coefficients from each RGB channel of the image
        # cHr, cVr, cDr represet the wavelet coefficients corresponding to the horizontal (cHr), vertical (cVr), and diagonal (cDr) components
        # obtained from the wavelet transform of the red channel
        proc_r = img[:, :, 0], (cHr, cVr, cDr)  # extracts pixel values only from the red channel of the image
        proc_g = img[:, :, 1], (cHg, cVg, cDg)  # extracts pixel values only from the green channel of the image
        proc_b = img[:, :, 2], (cHb, cVb, cDb)  # extracts pixel values only from the blue channel of the image

        # Apply inverse transform to each channel of the processed image, generating the steganographic image
        # It reconstructs the image data from the wavelet coefficients, thereby reversing the process of wavelet decomposition
        processed_rgbr = pywt.idwt2(proc_r, wavelet)
        processed_rgbg = pywt.idwt2(proc_g, wavelet)
        processed_rgbb = pywt.idwt2(proc_b, wavelet)
        
        # The three reconstructed color channels are merged into a single multi-channel image matrix
        finalimg = cv2.merge((processed_rgbr.astype(int), processed_rgbg.astype(int), processed_rgbb.astype(int)))

        # Plot steganographic image
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.title("Steganographic Image")
        plt.imshow(finalimg, aspect="equal")
        plt.show()
        
        # --------------------------------------------
        # Decoding Process
        # Steganography reversal process
        # ---------------------------------------------

        # Check if the password is correct before proceeding with the decoding process
        entered_password = input(f"\nEnter the password for {patient_choice}: ")

        if validate_password(patient_choice, entered_password, patients_data):

            # Apply the decoding transform to each channel of the stego image
            # applying dwt to 3 stego channel images to get coeffs of stego image in r,g,b
            # Decompose the image data into approximations (low-frequency) and detail coefficients (high-frequency)
            Psend_r = pywt.dwt2(processed_rgbr, wavelet)
            PcAr, (PcHr, PcVr, PcDr) = Psend_r # Get the approximate, horizontal, vertical and diagonal detail coefficients for red channel as a tuple where data may be embedded
            Psend_g = pywt.dwt2(processed_rgbg, wavelet)
            PcAg, (PcHg, PcVg, PcDg) = Psend_g # Get the approximate, horizontal, vertical and diagonal detail coefficients for green channel as a tuple where data may be embedded
            Psend_b = pywt.dwt2(processed_rgbb, wavelet)
            PcAb, (PcHb, PcVb, PcDb) = Psend_b # Get the approximate, horizontal, vertical and diagonal detail coefficients for blue channel as a tuple where data may be embedded

            dwt_labels = ["Approximation", "Horizontal Detail", "Vertical Detail", "Diagonal Detail"]

            plt.figure(figsize=(10, 10))

            #plotting the coefficient details
            show_coefficients_subplot([PcAr, PcHr, PcVr, PcDr], 5, dwt_labels)

        
            # Perform Singular Value Decomposition (SVD) on the stego image
            # PP: Left singular value matrix vectors 
            # PD: Singular values 1-D array that describes the "importance" or "strength" of each corresponding singular vector in P and Q
            # PQ: Right singular value matrix vectors
            # full_matrices = False gives compact form of SVD for memory efficiency
            PPr, PDr, PQr = np.linalg.svd(PcAr, full_matrices=False)
            PPg, PDg, PQg = np.linalg.svd(PcAg, full_matrices=False)
            PPb, PDb, PQb = np.linalg.svd(PcAb, full_matrices=False)

            # Reverse the information embedded in the 'D' parameter of the cover image that was performed in encoding process
            # Subtract from R,G,B channels values of cover image
            Decode_r = (PDr - Dr) / 0.1
            Decode_g = (PDg - Dg) / 0.1
            Decode_b = (PDb - Db) / 0.1

            # Combine the approximations with the hidden SVD values to reconstruct the hidden image
            # Calculating normalized differences between the singular values obtained during the decoding phase and the singular values obtained during the encoding phase for the red, green, and blue channels, respectively.
            reimgr = np.dot(P1r * Decode_r, Q1r) 
            reimgg = np.dot(P1g * Decode_g, Q1g)
            reimgb = np.dot(P1b * Decode_b, Q1b)

            # Obtain the reconstructed hidden image, which consists of the color channels combined with the normalized SVD differences
            # Converting the float point numbers to integers for calculation for red, green and blue channels respectively
            d = reimgr.astype(int)
            e = reimgg.astype(int)
            f = reimgb.astype(int)
            merged_img = cv2.merge((d, e, f)) # To create a single color image by combining these three channels

            # Pair each colour channel's pixel values with its respective DWT coefficients in a tuple to perform IDWT 
            proc_r = merged_img[:, :, 0], (cHr1, cVr1, cDr1) # Extracts pixel values only from the red channel of the image
            proc_g = merged_img[:, :, 1], (cHg1, cVg1, cDg1) # Extracts pixel values only from the green channel of the image
            proc_b = merged_img[:, :, 2], (cHb1, cVb1, cDb1) # Extracts pixel values only from the blue channel of the image

            # Apply inverse transform to each channel of the image to generate the final hidden information image
            # Hidden stego images get high resolution r,g,b seperate images/channels using IDWT
            processed_rgbr = pywt.idwt2(proc_r, wavelet)
            processed_rgbg = pywt.idwt2(proc_g, wavelet)
            processed_rgbb = pywt.idwt2(proc_b, wavelet)

            # Converting float to int for cv2 module to handle merging of pixel values
            x1 = processed_rgbr.astype(int) # For red channel
            y1 = processed_rgbg.astype(int) # For green channel
            z1 = processed_rgbb.astype(int) # For blue channel

            # Combine different high resolution r,g,b to get hidden image 
            hidden_image = cv2.merge((x1, y1, z1))

            # Display the decoded hidden image
            plt.figure(figsize=(5, 5))

            plt.axis('off')
            plt.title("Decoded Hidden Data")
            plt.imshow(hidden_image, aspect="equal")
            plt.show()
            
            break
            
        else:
            print("Incorrect password. Access denied. Program Terminating")
            break
    else:
        print("Invalid patient choice. Try again...")