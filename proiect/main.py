import numpy as np
import cv2
from cv2 import aruco
import math
import wave
import pyaudio
from objloader_simple import OBJ

def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # Normalizarea vectorilor
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # Calcularea bazei ortonormale
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    # Matrice de proiecție 3D
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)

def render(img, obj, projection, model, color=False, scale=5):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        color = (80, 27, 211)
        cv2.fillConvexPoly(img, imgpts, color)
    return img

def play_audio(file_path):
    chunk = 1024
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

camera_parameters = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
obj = OBJ('proiect/models/cube.obj', swapyz=True)
frame = cv2.imread("proiect/images/marker_1.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

print("IDs detectate:", ids)
frame_points = frame.copy()

if len(corners) != 0:
    crn = np.array(corners)[0, 0]
    marker_id = ids[0, 0]
    c = [crn[:, 0].mean(), crn[:, 1].mean()]
    radius = 30
    thickness = 60

    frame_points = cv2.circle(frame_points, (int(c[0]), int(c[1])), radius, (200, 200, 0), thickness)
    frame_points = cv2.circle(frame_points, (int(crn[0, 0]), int(crn[0, 1])), radius, (0, 255, 0), thickness)
    frame_points = cv2.circle(frame_points, (int(crn[2, 0]), int(crn[2, 1])), radius, (0, 0, 255), thickness)

    print("Colțuri detectate:", crn)

    sourcePoints = np.float32([(0, 0), (0, frame.shape[1]), (frame.shape[0], frame.shape[1]), (frame.shape[0], 0)]).reshape(-1, 1, 2)
    destinationPoints = np.float32([(crn[0, 0], crn[0, 1]), (crn[1, 0], crn[1, 1]), (crn[2, 0], crn[2, 1]), (crn[3, 0], crn[3, 1])]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
    print("Homografie calculată:", homography)

    crns = np.float32([[0, 0], [0, frame.shape[0] - 1], [frame.shape[1] - 1, frame.shape[0] - 1], [frame.shape[1] - 1, 0]]).reshape(-1, 1, 2)
    transformedCorners = cv2.perspectiveTransform(crns, homography)
    print("Colțuri transformate:", transformedCorners)

    frame_final = cv2.polylines(frame, [np.int32(transformedCorners)], True, 255, 10, cv2.LINE_AA)

    projection = projection_matrix(camera_parameters, homography)
    print("Matrice de proiecție:", projection)

    frame_final = render(frame_final, obj, projection, gray, color=False, scale=5)

    play_audio('proiect/audio/test_audio.wav')
else:
    frame_final = frame

frame_points = cv2.resize(frame_points, (frame_points.shape[1] // 5, frame_points.shape[0] // 5))
frame_final = cv2.resize(frame_final, (frame_final.shape[1] // 5, frame_final.shape[0] // 5))

cv2.imshow("Marker points", frame_points)
cv2.imshow('Final', frame_final)
k = cv2.waitKey(0)

cv2.destroyAllWindows()
