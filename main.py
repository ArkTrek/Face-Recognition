import os, sys
import math
import cv2
import numpy as np
import face_recognition

def face_confidence(dist, threshold = 0.6):
    range = (1 - threshold)
    linear_val = (1.0 - dist) / (range * 2.0)
    
    if dist > threshold:
        return str(round(linear_val * 100, 2)) + "%"
    else:
        val = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + "%"
    
class FaceRecognition:
    face_location = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    process_current_frame = []
    
    def __init__(self):
        pass
    
    def encode_faces(self):
        path = "dataset"
        dataset = os.listdir(path)
        for image in dataset:
            face_images = face_recognition.load_image_file(f'dataset'/{image})
            face_encoding = face_recognition.face_encodings(face_images)[0]
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
            
        print(self.known_face_names)
        
    def run_recognition(self):
        vc = cv2.VideoCapture(0)
        
        if not vc.isOpened():
            sys.exit("Video source not found...")
        
        while True:
            ret, frame = vc.read()
            
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                self.face_location = face_recognition.face_locations(rgb_small_frame)
                self.face_location = face_recognition.face_locations(rgb_small_frame, self.face_location)
                
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"
                    
                    dist = face_recognition.distt(self.known_face_encodings, face_encoding)
                    match_index = np.argmin(dist)
                    
                    if matches[match_index]:
                        name = self.known_face_encodings[match_index]
                        confidence = face_confidence(dist[match_index])
                        
                    self.face_names.append(f'{name} ({confidence})')
                    
            for(top, right, bottom, left), name in zip(self.face_location, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
                cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0,0,255), -1)
                cv2.putText(frame, name, (left +6, bottom -6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
            cv2.imshow("Face Recognizer", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        vc.release()
        cv2.destroyAllWindows()
                
if __name__ == "__main__":
    fr = FaceRecognition()
    fr.run_recognition()