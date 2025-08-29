import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Define the intensity calculation function (mean of RGB)
def calculate_intensity_as_mean_rgb(red, green, blue):
    return (red + green + blue) / 3

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
    # Convert RGB to BGR for OpenCV if necessary (PIL is RGB, OpenCV is BGR)
    if img_np.shape[2] == 3: # Check if it's a color image
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_np

    # Calculate average RGB values from the entire image
    # IMPORTANT: For accurate results, a specific Region of Interest (ROI)
    # corresponding to the "well" should be used. This prototype averages
    # the entire image.
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
        if not ((A1 < intensity < A2) or (A2 < intensity < A1)):
            st.error(f"Intensity value ({intensity:.2f}) is outside the model's trained range ({min(A1, A2):.2f} to {max(A1, A2):.2f}). Cannot predict concentration accurately.")
        else:
            # Solve for x (log_concentration) given y (intensity)
            # y = A2 + (A1 - A2) / (1 + (x / x0)**p)
            # (y - A2) = (A1 - A2) / (1 + (x / x0)**p)
            # (1 + (x / x0)**p) = (A1 - A2) / (y - A2)
            # (x / x0)**p = ((A1 - A2) / (y - A2)) - 1
            # x / x0 = (((A1 - A2) / (y - A2)) - 1)**(1/p)
            # x = x0 * (((A1 - A2) / (y - A2)) - 1)**(1/p)

            term = ((A1 - A2) / (intensity - A2)) - 1
            if term <= 0:
                st.error("Cannot predict concentration. The calculated intensity is outside the model's effective range or due to numerical instability.")
            else:
                log_concentration = x0 * (term**(1/p))
                predicted_concentration = 10**log_concentration
                st.success(f"Predicted MUC1 Concentration: {predicted_concentration:.2f} ng/ml")
    except Exception as e:
        st.error(f"Error predicting concentration: {e}. This might happen if the intensity is outside the model's effective range or due to numerical instability.")
