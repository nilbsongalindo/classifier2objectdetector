import keras
import numpy as np
import imutils
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import argparse
import time
import cv2

"""
Image: The input image
step: Step size, which indicates how many pixels we are going to "skip" in both (x,y) directions.
ws: The window size defines the widtg and height (in pixels) of the window we are going to extract from our image.
"""

def sliding_window(image, step, ws):
    #Loop over rows
    for y in range(0, image.shape[0] - ws[1], step):
        #Loop over columns
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x,y, image[y:y + ws[1], x:x + ws[0]])


"""
Image: The input image or which we wish to generate multi-scale representations.
scale: Our scale factor controls how much the image is resized at each layer.
minSize: Controls the minimum size of an output image (layer of our pyramid).
"""

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    #Yield the original image
    yield image
    
    #Keep looping over the image pyramid
    
    while True:
        #Compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        #if the resized image does not meet the nupplied minimum size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image


#ROI in pixels
size = (250, 250)
#minimum probability to filter weak detections
min_conf = 0.9
#whether or not to show extra visualizations for debugging
visualize = 1

#Variables for the object detection procedure
width = 600
pyr_scale = 1.5
win_step = 8
roi_size = size
input_size = (224, 224)


# load our network weights from disk
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=True)



# load the input image from disk, resize it such that it has the
# has the supplied width, and then grab its dimensions
orig = cv2.imread('data/beijaflor.jpeg')
orig = imutils.resize(orig, width=width)
(H, W) = orig.shape[:2]


# initialize the image pyramid
pyramid = image_pyramid(orig, scale=pyr_scale, minSize=roi_size)

# initialize two lists, one to hold the ROIs generated from the image
# pyramid and sliding window, and another list used to store the
# (x, y)-coordinates of where the ROI was in the original image

rois = []
locs = []

# time how long it takes to loop over the image pyramid layers and
# sliding window locations
start = time.time()


#Loop over the image pyramid
for image in pyramid:
    # determine the scale factor between the *original* image
    # dimensions and the *current* layer of the pyramid
    scale = W / float(image.shape[1])
    
    #For each layer of the imagem pyramid, loop over the sliding window locations.
    for (x, y, roiOrig) in sliding_window(image, win_step, roi_size):
        #Scale the (x,y)-coordinates of the ROI with respect to the *original* imagem dimensions.
        x = int(x  * scale)
        y = int(y * scale)
        w = int(roi_size[0] * scale)
        h = int(roi_size[1] * scale)
        
        #Take the ROI and process it so we can leter classify the region.
        roi = cv2.resize(roiOrig, input_size)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        
        #update our list of ROIs and associated coordinates
        rois.append(roi)
        locs.append((x, y, x+w, y+h))

if visualize > 0:
    #clone the original image and then draw a bounding box sorrounding the current region
    clone = orig.copy()
    cv2.rectangle(clone, (x,y), (x+w, y+h), (0,255,0),2)
    
    cv2.imshow("Visualization", clone)
    cv2.imshow("ROI", roiOrig)
    cv2.waitKey(0)

#Show how long it took to loop over the image pyramid layers and sliding window locations.
end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(end - start))

#convert the ROIs to a numpy array
rois = np.array(rois, dtype="float32")

#classify each of the proposal ROIs using ResNet.
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))

# decode the predictions and initialize a dictionary which maps class
# labels (keys) to any ROIs associated with that label (values)
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}


#Loop over the predictions
for (i,p) in enumerate(preds):
    #grab the prediction information for the current ROI
    (imagenetID, label, prob) = p[0]

    #Filter out weak detections by ensuring the predicted probability is greater than the minimum probability.
    if prob >= min_conf:
        #grab the bounding box associated with the prediction and convert the coordinates.
        box = locs[1]

        #grab the list of predictions for the label and add the bounding box and prob to the list.
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L


#loop over the labels for each of detected objects in the image
for label in labels.keys():
    #clone the original image so that we can draw on it
    print("[INFO] showing results for '{}'".format(label))
    clone = orig.copy()

    #loop over all bounding boxes for the current label
    for (box, prob) in labels[label]:
        #draw the bounding box on the image.
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 255, 0), 2)

        # show the results *before* applying non-maxima suppression, then
        # clone the image again so we can display the results *after*
        # applying non-maxima suppression
        cv2.imshow("Before", clone)
        clone = orig.copy()

# extract the bounding boxes and associated prediction
# probabilities, then apply non-maxima suppression
boxes = np.array([p[0] for p in labels[label]])
proba = np.array([p[1] for p in labels[label]])
boxes = non_max_suppression(boxes, proba)

#loop over all bounding boxes
for (startX, startY, endX, endY) in boxes:
    #draw the bounding box and label on the image
    cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(clone, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


# show the output after apply non-maxima suppression
cv2.imshow("After", clone)
cv2.waitKey(0)
