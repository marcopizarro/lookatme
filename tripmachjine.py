import numpy as np
import cv2
import random

cap = cv2.VideoCapture(0)
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
leftCasc = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
rightCasc = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
mouthCasc = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
noseCasc = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

state = 1
ret, frame = cap.read()
(h, w) = frame.shape[:2]
red_img = np.full((h, w, 3), (0, 0, 255), np.uint8)
prev = frame
size = 640
count = 0
blank_image = np.zeros((size, size, 3), np.uint8)

Matrix = [[0 for x in range(10)] for y in range(10)]

for i in range(10):
    for j in range(10):
        Matrix[i][j] = random.randint(0, 3)
print(Matrix)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    xr, yr, wr, hr = 835, 407, 469, 469
    xrt, yrt, wrt, hrt = 835, 407, 469, 469
    xlt, ylt, wlt, hlt = 835, 407, 469, 469
    xm, ym, wm, hm = 835, 407, 469, 469
    xn, yn, wn, hn = 835, 407, 469, 469

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=20,
        minSize=(30, 30)
    )

    nose = noseCasc.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=20,
        minSize=(30, 30)
    )

    left = leftCasc.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=20,
        minSize=(30, 30)
    )

    mouth = mouthCasc.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=30,
        minSize=(30, 30)
    )

    right = rightCasc.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=20,
        minSize=(30, 30)
    )
    for (x, y, w, h) in mouth:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if x:
            (xm, ym, wm, hm) = mouth[0]
    for (x, y, w, h) in nose:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if x:
            (xn, yn, wn, hn) = nose[0]
    for (x, y, w, h) in left:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if x:
            (xlt, ylt, wlt, hlt) = left[0]
    for (x, y, w, h) in right:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if x:
            (xrt, yrt, wrt, hrt) = right[0]
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if x:
            (xr, yr, wr, hr) = faces[0]

    # blurred_image = cv2.GaussianBlur(frame, (7, 7), 0)
    canny2 = cv2.Canny(gray, 50, 150)
    canny2 = cv2.cvtColor(canny2, cv2.COLOR_GRAY2RGB)
    canny2 = cv2.resize(canny2, (size, size))
    choices = []

    # added_image = cv2.addWeighted(canny2, 0.4, frame, 0.1, 0)

    # print(xr, yr, wr, hr)
    # cropped_image = frame[xr:yr, xr+wr:yr+hr]
    facecropped = frame[yr:yr+hr, xr:xr+wr]
    facecropped = cv2.resize(facecropped, (size, size))

    nosecrop = frame[yn:yn+hn, xn:xn+wn]
    nosecrop = cv2.resize(nosecrop, (size, size))

    mouthcropped = frame[ym:ym+hm, xm:xm+wm]
    mouthcropped = cv2.resize(mouthcropped, (size, size))

    leftcropped = frame[ylt:ylt+hlt, xlt:xlt+wlt]
    leftcropped = cv2.resize(leftcropped, (size, size))

    rightcropped = frame[yrt:yrt+hrt, xrt:xrt+wrt]
    rightcropped = cv2.resize(rightcropped, (size, size))

    frame = cv2.resize(frame, (size, size))

    choices = [nosecrop, rightcropped, mouthcropped, leftcropped]
    if state == 0:
        concat = cv2.hconcat([blank_image, nosecrop, blank_image])
        concat2 = cv2.hconcat([leftcropped, mouthcropped, rightcropped])
        concat3 = cv2.vconcat([concat, concat2])
        cv2.imshow('concat', concat3)
    elif state == 1:
        row0 = cv2.hconcat([choices[Matrix[0][1]], choices[Matrix[0][2]], choices[Matrix[0][3]], choices[Matrix[0][4]],
                            choices[Matrix[0][5]], choices[Matrix[0][6]], choices[Matrix[0][7]], choices[Matrix[0][8]], choices[Matrix[0][9]]])
        row1 = cv2.hconcat([choices[Matrix[1][1]], choices[Matrix[1][2]], choices[Matrix[1][3]], choices[Matrix[1][4]],
                            choices[Matrix[1][5]], choices[Matrix[1][6]], choices[Matrix[1][7]], choices[Matrix[1][8]], choices[Matrix[1][9]]])
        row2 = cv2.hconcat([choices[Matrix[2][1]], choices[Matrix[2][2]], choices[Matrix[2][3]], choices[Matrix[2][4]],
                            choices[Matrix[2][5]], choices[Matrix[2][6]], choices[Matrix[2][7]], choices[Matrix[2][8]], choices[Matrix[2][9]]])
        row3 = cv2.hconcat([choices[Matrix[3][1]], choices[Matrix[3][2]], choices[Matrix[3][3]], choices[Matrix[3][4]],
                            choices[Matrix[3][5]], choices[Matrix[3][6]], choices[Matrix[3][7]], choices[Matrix[3][8]], choices[Matrix[3][9]]])
        row4 = cv2.hconcat([choices[Matrix[4][1]], choices[Matrix[4][2]], choices[Matrix[4][3]], choices[Matrix[4][4]],
                            choices[Matrix[4][5]], choices[Matrix[4][6]], choices[Matrix[4][7]], choices[Matrix[4][8]], choices[Matrix[4][9]]])
        row5 = cv2.hconcat([choices[Matrix[5][1]], choices[Matrix[5][2]], choices[Matrix[5][3]], choices[Matrix[5][4]],
                            choices[Matrix[5][5]], choices[Matrix[5][6]], choices[Matrix[5][7]], choices[Matrix[5][8]], choices[Matrix[5][9]]])

        all = cv2.vconcat([row0, row1, row2, row3, row4, row5])

        cv2.imshow('concat', all)

    else:
        ret, frame = cap.read()
        (h, w) = frame.shape[:2]
        print(w)
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), count, 1.0)
        warp = cv2.warpAffine(frame, M, (w, h))
        red_img = np.full((h, w, 3), (random.randint(
            0, 255), random.randint(0, 255), random.randint(0, 255)), np.uint8)

        warp = cv2.addWeighted(warp, 0.8, red_img, 0.2, 0)

        count += 2

        # warp = cv2.resize(warp, (size, size))
        # prev = cv2.resize(prev, (size, size))

        dst = cv2.addWeighted(prev, 0.9, warp, .1, 0.0)
        prev = dst
        cv2.imshow('concat', dst)
    cv2.setWindowProperty(
        "concat", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #

    # cv2.imshow('crop', cropped_image)

    if cv2.waitKey(2) & 0xFF == ord('r'):
        if state <= 1:
            state += 1
        else:
            state = 0
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Turn on Laptop's webcam
# cap = cv2.VideoCapture(0)

# while True:

#     ret, frame = cap.read()

#     # Locate points of the documents
#     # or object which you want to transform
#     pts1 = np.float32([[0, 260], [640, 260],
#                        [0, 400], [640, 400]])
#     pts2 = np.float32([[0, 0], [400, 0],
#                        [0, 640], [400, 640]])

#     # Apply Perspective Transform Algorithm
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     result = cv2.warpPerspective(frame, matrix, (1000, 1000))

#     # Wrap the transformed image
#     # cv2.imshow('frame', frame)  # Initial Capture
#     cv2.imshow('frame1', result)  # Transformed Capture

#     if cv2.waitKey(24) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
