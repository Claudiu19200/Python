import cv2
import numpy as np

# Crează un dicționar de markeri ArUco
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Alege un ID de marker
marker_id = 42  # Poți alege orice ID între 0 și 249

# Generează markerul ArUco
marker = np.zeros((200, 200), dtype=np.uint8)
marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 200, marker, 1)

# Salvează markerul ca imagine
cv2.imwrite('proiect/images/marker_1.png', marker)

# Afișează imaginea markerului
cv2.imshow('Marker', marker)
cv2.waitKey(0)
cv2.destroyAllWindows()
