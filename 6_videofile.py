import cv2

videoCapture = cv2.VideoCapture("./videos/Wildlife.mp4")
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print fps
# uncompressed	YUVencoding produces larges files
videoWriterYUV = cv2.VideoWriter("./videos/OutputVidYUV.avi",
                                 cv2.VideoWriter_fourcc("I", "4", "2", "0"),
                                 fps,
                                 size)
videoWriterMPEG1 = cv2.VideoWriter("./videos/OutputVidMPEG1.avi",
                                   cv2.VideoWriter_fourcc("P", "I", "M", "1"),
                                   fps,
                                   size)
videoWriterMPEG4 = cv2.VideoWriter("./videos/OutputVidMPEG4.avi",
                                   cv2.VideoWriter_fourcc("X", "V", "I", "D"), # "DIVX"
                                   fps,
                                   size)


while True:
    success, frame = videoCapture.read()
    if cv2.waitKey(1) == 27 or not success:  # For a	set	of cameras use grab
        break  # esc to quit
    cv2.imshow('MyWindow', frame)
    videoWriterYUV.write(frame)
    videoWriterMPEG1.write(frame)
    videoWriterMPEG4.write(frame)
    success, frame = videoCapture.read()
videoCapture.release()
cv2.destroyAllWindows()
