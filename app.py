import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Define the intensity calculation function (using blue channel)
def calculate_intensity_as_mean_rgb(red, green, blue):
    return blue

# Define the logistic function (using the fitted parameters from CEJ-S-25-22426.pdf)
def logistic(x, A1, A2, x0, p):
    return A2 + (A1 - A2) / (1 + (x / x0)**p)

# Hardcoded fitted parameters from CEJ-S-25-22426.pdf (Figure 5a table)
logistic_params = {
    'A1': 30.02861,
    'A2': 97.39135,
    'x0': 3.13541,
    'p': 12.37969
}

st.title("MUC1 Concentration Predictor")
st.write("Capture or upload an image of the well to predict MUC1 concentration.")

# Add camera input
camera_image = st.camera_input("Take a picture")

# Add file uploader (optional, can be removed if only camera is desired)
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

# Determine which input to use
input_image = camera_image if camera_image is not None else uploaded_file

if input_image is not None:
    # Display the uploaded image
    image = Image.open(input_image)
    st.image(image, caption='Input Image', use_column_width=True)
    st.write("")
    st.write("Predicting concentration...")

    # Convert PIL Image to OpenCV format
    img_np = np.array(image)
    # Convert to BGR for OpenCV, handling RGBA if present
    if img_np.shape[2] == 4: # If it's RGBA
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    elif img_np.shape[2] == 3: # If it's RGB
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else: # Grayscale or other (should ideally be 3 or 4 channels for this app)
        img_cv = img_np

    # Ensure img_cv is 3-channel for consistency in ROI processing
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)

    # Function to find and extract the circular ROI
    def find_and_extract_roi(img_cv):
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Parameters for HoughCircles might need tuning
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=100, param2=30, minRadius=20, maxRadius=200)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Assuming the largest circle is the desired ROI
            circles_sorted = sorted(circles[0, :], key=lambda x: x[2], reverse=True)
            x, y, r = circles_sorted[0]

            # Make the ROI a bit smaller
            scale_factor = 0.65 # Adjust this value to make ROI even smaller
            r = int(r * scale_factor)
            r = max(1, r) # Ensure radius is at least 1 to avoid errors

            mask = np.zeros(img_cv.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            roi = cv2.bitwise_and(img_cv, mask)

            # img_with_circle = img_cv.copy()
            # cv2.circle(img_with_circle, (x, y), r, (0, 255, 0), 2)
            # cv2.circle(img_with_circle, (x, y), 2, (0, 0, 255), 3)

            # st.image(img_with_circle, caption='Detected Circle', use_column_width=True)
            st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption='Extracted ROI', use_column_width=True)

            return roi
        else:
            st.warning("No circular region detected. Proceeding with entire image for intensity calculation.")
            return img_cv

    # Find and extract ROI
    roi_image = find_and_extract_roi(img_cv)

    # Calculate average RGB values from the ROI
    # Filter out black pixels (0,0,0) which are outside the circular ROI
    non_black_pixels = roi_image[np.all(roi_image != [0, 0, 0], axis=-1)]

    if non_black_pixels.size > 0:
        avg_bgr = np.mean(non_black_pixels, axis=0)
    else:
        st.warning("ROI is entirely black or no non-black pixels found. Using entire image average.")
        avg_bgr = np.mean(img_cv, axis=(0, 1))
    avg_blue, avg_green, avg_red = avg_bgr[0], avg_bgr[1], avg_bgr[2]

    # Calculate intensity using the mean of RGB values
    intensity = calculate_intensity_as_mean_rgb(avg_red, avg_green, avg_blue)
    st.write(f"Calculated Intensity (Mean RGB): {intensity:.2f}")

    # Predict concentration using the logistic model
    # The model maps LOG(concentration) to direct intensity (mean RGB)

    A1 = logistic_params['A1']
    A2 = logistic_params['A2']
    x0 = logistic_params['x0']
    p = logistic_params['p']

    try:
        # Ensure intensity is within the valid range for the logistic function
        # The logistic function is defined for y values between A1 and A2 (or A2 and A1)
        # Handle cases where intensity is outside the model's primary prediction range
        if intensity >= max(A1, A2):
            predicted_concentration = 0.0
            st.success(f"Predicted MUC1 Concentration: {predicted_concentration:.2f} ng/ml (Intensity too high, indicating zero concentration).")
        elif intensity <= min(A1, A2):
            # This case implies very high concentration, potentially beyond the model's fitted range.
            # The logistic function's inverse might become numerically unstable or undefined here.
            # For now, we'll try to calculate, but if it fails, we'll provide a specific message.
            try:
                term = ((A1 - A2) / (intensity - A2)) - 1
                if term <= 0:
                    # This happens if intensity is very close to A2 or outside the valid range for the inverse calculation
                    st.error("Cannot predict concentration for this low intensity. It's outside the model's effective range for a valid calculation.")
                else:
                    log_concentration = x0 * (term**(1/p))
                    predicted_concentration = 10**log_concentration
                    st.success(f"Predicted MUC1 Concentration: {predicted_concentration:.2f} ng/ml")
            except Exception as e:
                st.error(f"Error predicting concentration for very low intensity: {e}. This might happen if the intensity is outside the model's effective range or due to numerical instability.")
        else:
            # Intensity is within the valid range (A1, A2)
            term = ((A1 - A2) / (intensity - A2)) - 1
            if term <= 0:
                st.error("Cannot predict concentration. The calculated intensity is outside the model's effective range or due to numerical instability.")
            else:
                log_concentration = x0 * (term**(1/p))
                predicted_concentration = 10**log_concentration
                st.success(f"Predicted MUC1 Concentration: {predicted_concentration:.2f} ng/ml")
    except Exception as e:
        st.error(f"Error predicting concentration: {e}. This might happen if the intensity is outside the model's effective range or due to numerical instability.")
