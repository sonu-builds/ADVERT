import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    tip_ids = [4, 8, 12, 16, 20]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        text = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                lm_list = []
                h, w, c = frame.shape

                for id, lm in enumerate(hand_landmarks.landmark):
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                if lm_list:
                    fingers = []

                    # Thumb
                    if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0]-1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Other fingers
                    for i in range(1, 5):
                        if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i]-2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # Gesture → Text mapping
                    if fingers == [0,1,0,0,0]:
                        text = "A"
                    elif fingers == [0,1,1,0,0]:
                        text = "B"
                    elif fingers == [0,1,1,1,0]:
                        text = "C"
                    elif fingers == [0,1,1,1,1]:
                        text = "D"
                    elif fingers == [1,1,1,1,1]:
                        text = "E"
                    else:
                        text = "?"

        cv2.putText(frame, f'Text: {text}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Sign Language to Text", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()