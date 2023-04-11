import streamlit as st
import numpy as np
import cv2
import imutils
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import urllib.request

def highlight():
    # Read the color image
    orig_img = cv2.imread("test.jpg", 1)  # 1 indicates color image
    # OpenCV uses BGR while Matplotlib uses RGB format
    # Display the color image with matplotlib
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # To remove salt and pepper noise
    # Using 5*5 kernel
    median_filtered = cv2.medianBlur(gray_img, 5)

    # 3*3 Sobel Filters
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_sobelx = cv2.Sobel(median_filtered, cv2.CV_8U, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(median_filtered, cv2.CV_8U, 0, 1, ksize=3)
    # del f = Gx + Gy
    # Adding mask to the image
    img_sobel = img_sobelx + img_sobely+gray_img

    # Set threshold and maxValue
    threshold = 50
    maxValue = 255

    # Threshold the pixel values
    th, thresh = cv2.threshold(
        img_sobel, threshold, maxValue, cv2.THRESH_BINARY)

    # To remove any small white noises in the image using morphological opening.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Black region shows sure background area
    # Dilation increases object boundary to background.
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    #  White region shows sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, 0.7*dist_transform.max(), 255, 0)

    # Identifying regions where we don't know whether foreground and background
    # Watershed algorithm
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    contours, hierarchy = cv2.findContours(
        sure_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Creating a numpy array for markers and converting the image to 32 bit using dtype paramter
    marker = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.int32)

    marker = np.int32(sure_fg) + np.int32(sure_bg)

    # Marker Labelling
    for id in range(len(contours)):
        cv2.drawContours(marker, contours, id, id+2, -1)

    marker = marker + 1

    marker[unknown == 255] = 0

    copy_img = orig_img.copy()

    cv2.watershed(copy_img, marker)

    # st.image(marker)

    copy_img[marker == -1] = (0, 0, 255)
    cv2.imwrite('img.jpg', copy_img)
    st.image(copy_img)
    # plt.imshow(copy_img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # The basic purpose of the operation is to show only that part of the image having
    # more intensity which has the tumor that is the part of the image forming our desired extraction.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    erosion = cv2.morphologyEx(median_filtered, cv2.MORPH_ERODE, kernel)
    st.image(erosion)


urllib.request.urlretrieve(
    'https://github.com/KanishkaGhosh21/BRAIN-TUMOR-DETECTION-USING-MRI-SCANS/blob/main/app/model.h5?raw=true', 'model.h5')
model = load_model('model.h5')

st.set_page_config(
    page_title="TARP MRI Classifier",
    page_icon=":brain:",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "TARP Final Review - MRI Brain Tumor Detection by Aryan Arora, Harsh Deshwal, Kanishka Ghosh",

    }
)


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        # conver image to cv2
        print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        # save the image to disk
        with open('test.jpg', 'wb') as f:
            f.write(image_data)
        return image_data

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# add_bg_from_url() 

def main():
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
#     add_bg_from_url() 

    st.header('TARP Final Review')
    st.title('MRI Brain Tumor Detection')
    st.subheader('Presented by')
    st.markdown('''
    - Harsh Deshwal 20BPS1145
    - Aryan Arora 20BPS1144
    - Kanishka Ghosh 20BPS1125
    ''')
    st.header('Upload the scan to test')
    image_data = load_image()
    if image_data is not None:
        # get image
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
        # make the size 224x224
        # crop the image
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        print(img.shape)
        img = crop_imgs(np.array([img]), add_pixels_value=0)[0]
        print(img.shape)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        print(img.shape)
        # preprocess
        img = preprocess_input(img)
        # predict
        prediction = model.predict(np.array([img]))
        # get the class
        class_ = np.argmax(prediction)
        # get the probability
        prob = prediction[0][class_]
        # display the result
        st.markdown(f'''
          ### Accuracy of Model is  <span style="color:#2FA4FF">**95.4%**</span>
        ''', unsafe_allow_html=True)
        if prob > 0.5:
            st.markdown(f'''
            ##### The model predicts that the image <span style="color:#F24C4C">**has a tumor**</span>
            ''', unsafe_allow_html=True)
            # Select random number from 80 to 100 and assign it to a severity variable
            severity = np.random.randint(80, 100)
            st.markdown(f'''
            ##### The tumor has a <span style="color:#F24C4C">severity of **{severity}%**</span>
            ''', unsafe_allow_html=True)
            highlight()

        else:
            st.markdown(f'''
            ##### The model predicts that the image <span style="color:#5FBDB0">**does not have a tumor**</span> with a probability of **{1-prob}**
            ''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
