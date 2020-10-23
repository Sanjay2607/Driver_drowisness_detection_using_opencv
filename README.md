# Driver_drowisness-_detection_using_opencv

Face landmarks are used to localize and represent salient regions of the face, such as Eyes, Eyebrows, Nose, Mouth, Jawline. The face Shape predictor file gives 68 points of the facial landmark with [x, y] coordinates. Detecting facial landmarks is done in two steps: Localize the face in the image. Detect the key facial structures on the face ROI.

HOG + Linear SVM Algorithm for object landmark detection:
http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/

Dlib’s facial landmark detector The pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates that map to facial structures on the face. The indexes of the 68 coordinates can be visualized on the image below:
![facial landmaRK](https://raw.githubusercontent.com/Lalit-ai/driver_drowsiness_raspberry_pi/master/facial_landmarks_68markup-768x619.jpg)

Download Dlib Face shape predictor 68 landmarks .dat file Download From: https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
