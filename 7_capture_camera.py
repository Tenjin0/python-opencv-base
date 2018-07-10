import cv2

cameraCapture = cv2.VideoCapture(0)

fps = 30
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# videoWriter = cv2.VideoWriter(
#     './videos/MyOutputVid.avi',	cv2.VideoWriter_fourcc('I', '4', '2', '0'),	fps, size)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('./videos/output.avi', fourcc, 20.0, (640, 480))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./videos/outputCamera.avi', fourcc, 20.0, size)

numFrameRemaining = 5 * fps - 1
while cameraCapture.isOpened():
    success, frame = cameraCapture.read()
    # if success:
    #     frame = cv2.flip(frame, 0)

    # write the flipped frame
    out.write(frame)

    cv2.imshow('MyWindow',	frame)
    numFrameRemaining -= 1
    # print(success, numFrameRemaining)
    if cv2.waitKey(1) == 27 or numFrameRemaining <= 0:
        break  # esc to quit
# while True:
#         ret_val, img = cameraCapture.read()
#         cv2.imshow('my webcam', img)
#         if cv2.waitKey(1) == 27:
#             break  # esc to quit
cameraCapture.release()
out.release()
cv2.destroyAllWindows()
