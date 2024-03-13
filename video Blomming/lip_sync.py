import face_recognition
import cv2

"""

Je ne me rappelle plus comment installer face_recognition
Les fonctions de face recognition sont sur :
https://face-recognition.readthedocs.io/

ceci est un exercice de reconnaissance faciale
J'ai telechargé une vidéo youtube par le biais de ytdl:

https://pypi.org/project/yt-dlp/

https://www.youtube.com/watch?v=rSpUn98R20A pour Runnin' de Thugshell





Cédric,
Le 19/11/2023.



"""

# Open the input movie file
input_movie = cv2.VideoCapture("e:/Computer vision/face_recognition/face_recognition-master/vidz/blooming.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('e:/Computer vision/face_recognition/face_recognition-master/vidz/blomming_lips_600.avi', fourcc, 29.97, (800, 600))

# Load some sample pictures and learn how to recognize them.

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1   
    #if frame_number <400 and frame_number>300:
    # Quit when the input video file ends
    if frame_number>600:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    #face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    
    if len(face_landmarks_list)>0:
        face_landmarks_list= face_landmarks_list[0]
        top_lip= face_landmarks_list["top_lip"]
        bottom_lip= face_landmarks_list["bottom_lip"]
        print( "moaaaa " +str(top_lip))
        for i in top_lip:
            cv2.circle(frame,i, 1, (0,0,255), -1)
        for i in bottom_lip:
            cv2.circle(frame,i, 1, (0,0,255), -1)
        cv2.circle(frame,top_lip[3], 2, (0,255,0), -1)      
        cv2.circle(frame,bottom_lip[3], 2, (0,255,0), -1)        

    face_names = []

    

    # Label the results
    for (top, right, bottom, left) in face_locations:

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

        # Draw a label with a name below the face
        
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
