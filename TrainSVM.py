import face_recognition
import imutils.paths as paths
import cv2
import os
import pickle

dataset = "C:\\Users\\Nghia\\desktop\\data\\Data\\"
module = "C:\\Users\\Nghia\\desktop\\data\\encodingSVM.pickle"
# Training the SVC classifier

knownEncodings = []
knownNames = []

imagepaths = list(paths.list_images(dataset))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagepaths):
    print("[INFO] processing image {}/{}".format(i + 1,len(imagepaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model= "hog")
    encodings = face_recognition.face_encodings(rgb, boxes)[0]
    print(encodings)
    knownEncodings.append(encodings)
    knownNames.append(name)
    print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
output = open(module, "wb")
pickle.dump(data, output)
output.close()