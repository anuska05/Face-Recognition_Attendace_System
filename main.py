import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)  # 0 for 1st webcam and video will be available through webcam
# Load Known Faces
# loads an image file
anu_image = face_recognition.load_image_file("Faces/Anuska.jpg")
# image is converted to numbers so that it is easier to compare; [0] refers to 1st face
anu_encoding = face_recognition.face_encodings(anu_image)[0]
rohan_image = face_recognition.load_image_file("Faces/Rohan.jpeg")
rohan_encoding = face_recognition.face_encodings(rohan_image)[0]

known_face_encodings = [anu_encoding, rohan_encoding]
known_face_names = ["Anuska", "Rohan"]

# list of expected students
students = known_face_names.copy()
face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces (face_locations()- recognizes faces and returns bounding boxes)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Add the text if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                        lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
