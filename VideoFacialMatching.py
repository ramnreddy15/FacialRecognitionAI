#import OpenCV module
import cv2

#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as
#it is needed by OpenCV face recognizers
import numpy as np
####################
#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Venkat Nusum", "Ram Reddy", "Ramya Reddy"]

################
# function to detect face using OpenCV
def detect_face(img):
    print("Detect Face 1 ...: ")
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Detect Face 2 ...: ")
    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    # face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

    print("Detect Face 3 ...: ")
    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    #faces = face_cascade.detectMultiScale(gray);
    print("Detect Face 4 ...: ")
    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]

###############
# In[4]:

# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    print("Preparing data... step-1")
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:
        print("Preparing data... dir_name: ", dir_name)
        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:
            print("image_name: ", image_name)
            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)
            print("Read image_name: ", image_path)
            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(1)
            print("Training on image...: ", image_path)
            # detect face
            face, rect = detect_face(image)
            print("Face is Detected...: ", image_path)
            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                print("Face is Detected...: ", image_path)
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

############################

# In[5]:

#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

##############################
# In[6]:

#create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
###############################
# In[7]:

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))
################################
# In[8]:

# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

############################
# In[9]:

# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)

    print("Confidence: ", round(confidence))

    if(confidence > 50):
        # get name of respective label returned by face recognizer
        label_text = subjects[label]
        print("label_text: ", label_text)
    else:
        label_text = "Unknown"

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img
############################
# In[10]:

print("Predicting images...")

# load test images
#test_img1 = cv2.imread("test-data/test1.jpg")
#test_img2 = cv2.imread("test-data/test2.jpg")

# perform a prediction
#predicted_img1 = predict(test_img1)
#predicted_img2 = predict(test_img2)

print("Prediction complete")

# display both images
#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))

#begin
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #, flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        frame = predict(frame)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()

#end

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
##############################

