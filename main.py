import time
from pathlib import Path

import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key

#https://www.coolmathgames.com/0-fireboy-and-water-girl-in-the-forest-temple?variant=next

def recognize_gestures(hand):
    """Trả về các hành động của một bàn tay: nhảy và/hoặc di chuyển."""
    actions = []

    # Mỗi cặp là (đầu ngón, khớp giữa). Đầu ngón cao hơn khớp => ngón duỗi.
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = 0
    for tip, joint in fingers:
        if hand[tip].y < hand[joint].y:
            extended += 1

    if extended == 4:
        actions.append("JUMP")

    thumb_dx = hand[4].x - hand[2].x
    thumb_dy = abs(hand[4].y - hand[2].y)

    if abs(thumb_dx) > 0.7 * thumb_dy:
        if thumb_dx > 0:
            actions.append("RIGHT")
        else:
            actions.append("LEFT")

    return actions


def main():
    model_path = Path(__file__).with_name("hand_landmarker.task")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    # Mỗi nhân vật dùng một nhóm phím riêng trong game.
    key_map = {
        "Player 1": {
            "LEFT": Key.left,
            "RIGHT": Key.right,
            "JUMP": Key.up,
        },
        "Player 2": {
            "LEFT": "a",
            "RIGHT": "d",
            "JUMP": "w",
        },
    }
    keyboard = Controller()
    held_keys = set()
    last_timestamp = 0

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Không mở được webcam")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                # Lật ảnh như gương: bên trái hình là vùng của Player 1.
                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape

                # OpenCV dùng BGR, còn MediaPipe nhận RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Chế độ VIDEO cần thời gian tăng dần cho từng khung hình.
                timestamp = int(time.monotonic() * 1000)
                if timestamp <= last_timestamp:
                    timestamp = last_timestamp + 1
                last_timestamp = timestamp
                result = detector.detect_for_video(mp_image, timestamp)

                wanted_keys = set()
                cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

                for hand in result.hand_landmarks:
                    total_x = 0
                    total_y = 0
                    for point in hand:
                        total_x += point.x
                        total_y += point.y

                    center_x = total_x / len(hand)
                    center_y = total_y / len(hand)

                    if center_x < 0.5:
                        player = "Player 1"
                    else:
                        player = "Player 2"

                    gestures = recognize_gestures(hand)
                    for gesture in gestures:
                        wanted_keys.add(key_map[player][gesture])

                    # Vẽ 21 điểm mốc và nhãn hành động lên hình webcam.
                    for point in hand:
                        x = int(point.x * width)
                        y = int(point.y * height)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    if gestures:
                        gesture_text = " + ".join(gestures)
                    else:
                        gesture_text = "NONE"
                    cv2.putText(
                        frame,
                        f"{player}: {gesture_text}",
                        (int(center_x * width), int(center_y * height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                # Chỉ gửi sự thay đổi, tránh nhấn lại phím ở mỗi khung hình.
                for key in held_keys:
                    if key not in wanted_keys:
                        keyboard.release(key)

                for key in wanted_keys:
                    if key not in held_keys:
                        keyboard.press(key)
                held_keys = wanted_keys

                cv2.imshow("Hand tracking", frame)
                # waitKey trả về mã của phím vừa bấm; 0xFF lấy phần mã phím cần dùng.
                pressed_key = cv2.waitKey(1)
                if pressed_key & 0xFF == ord("q"):
                    break
    finally:
        for key in held_keys:
            keyboard.release(key)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
