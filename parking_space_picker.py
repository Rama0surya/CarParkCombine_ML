import cv2
import pickle
import math

size1 = (90, 25)
size2 = (25, 90)  # Ukuran tempat parkir kedua

pt1_x, pt1_y, pt2_x, pt2_y = None, None, None, None
selected_size = size1  # Ukuran tempat parkir yang dipilih
delete_mode = False  # Mode untuk menghapus picker

try:
    with open('park_positions', 'rb') as f:
        park_positions = pickle.load(f)
except FileNotFoundError:
    park_positions = []

def parking_line_counter():
    line_count = int((math.sqrt((pt2_x - pt1_x) ** 2 + (pt2_y - pt1_y) ** 2)) / selected_size[1])
    return line_count

def mouse_events(event, x, y, flag, param):
    global pt1_x, pt1_y, pt2_x, pt2_y, delete_mode, park_positions

    if event == cv2.EVENT_LBUTTONDOWN:
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        pt2_x, pt2_y = x, y
        if not delete_mode:
            parking_spaces = parking_line_counter()
            if parking_spaces == 0:
                park_positions.append((len(park_positions), x, y, selected_size[0], selected_size[1]))
            else:
                for i in range(parking_spaces):
                    park_positions.append((len(park_positions), pt1_x, pt1_y + i * selected_size[1], selected_size[0], selected_size[1]))

    elif event == cv2.EVENT_RBUTTONDOWN:
        delete_mode = True

    with open('park_positions', 'wb') as f:
        pickle.dump(park_positions, f)

while True:
    img = cv2.imread('example_image.png')

    for position in park_positions:
        if len(position) == 5:  # Pastikan tuple memiliki jumlah elemen yang sesuai
            id, x, y, w, h = position
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(img, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_events)

    key = cv2.waitKey(30)
    if key == 27:
        break
    elif key == ord('1'):
        selected_size = size1
        print("Selected Size: 1")
    elif key == ord('2'):
        selected_size = size2
        print("Selected Size: 2")
    elif key == ord('d'):
        if park_positions:
            park_positions.pop()
            delete_mode = False

cv2.destroyAllWindows()
