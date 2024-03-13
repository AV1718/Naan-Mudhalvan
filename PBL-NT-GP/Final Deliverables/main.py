import sys
import cv2
from imutils import face_utils
import dlib
from scipy.spatial import distance

import ibmiotf.application
import ibmiotf.device
import random

# Provide your IBM Watson Device Credentials
organization = "orgid"
deviceType = "devtype"
deviceId = "devid"
authMethod = "token"
authToken = "12345678"


def ibmstart(x):

    def myCommandCallback(cmd):
        print("Command received: %s" % cmd.data['command'])
        print(cmd)

    try:
        deviceOptions = {"org": organization, "type": deviceType,
                         "id": deviceId, "auth-method": authMethod, "auth-token": authToken}
        deviceCli = ibmiotf.device.Client(deviceOptions)
        # ..............................................

    except Exception as e:
        print("Caught exception connecting device: %s" % str(e))
        sys.exit()

    deviceCli.connect()
    data = {'Status': x}
    # print data

    def myOnPublishCallback():
        print("Published Status = %s" % x, "to IBM Watson")

    success = deviceCli.publishEvent(
        "DD", "json", data, qos=0, on_publish=myOnPublishCallback)
    if not success:
        print("Not connected to IoTF")

    deviceCli.commandCallback = myCommandCallback
    deviceCli.disconnect()

# Function to calculate the eye aspect ratio (EAR)


def eye_aspect_ratio(eye):
    # Calculate the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Calculate the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# Constants for eye aspect ratio (EAR) thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Load the facial landmark predictor from dlib
# Path to the shape predictor model file
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize counters and drowsy status
frame_counter = 0
drowsy = False
lm = False

# Start the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture

    ret, frame = video_capture.read()
    if not ret:

        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    if len(faces) != 0:
        face = faces[0]

        # Predict the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Calculate the eye aspect ratios (EARs)
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EARs of both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the threshold
        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            # Reset the frame counter
            frame_counter = 0
            drowsy = False

        # Check if drowsiness is detected for consecutive frames
        if frame_counter >= EAR_CONSEC_FRAMES:
            drowsy = True

            cv2.putText(frame, "Drowsy!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if lm != drowsy:
            lm = drowsy
            ibmstart(drowsy)
        # Display the eye aspect ratio on the frame

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Drowsiness Detection", frame)

    # Quit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
