# 키를 입력받고 팔 길이, 다리 길이, 어깨 너비 측정 

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe의 Pose 모듈을 로드합니다.
mp_pose = mp.solutions.pose

# 이미지의 크기
image_width_px = 550
image_height_px = 550

# 사용자로부터 키를 입력 받습니다. (단위: cm)
height_cm = float(input("키를 입력하세요 (단위: cm): "))

# 이미지의 픽셀 당 길이 계산
px_per_cm = 480 / height_cm

# 이미지를 불러옵니다.
image_path = (r'이미지 경로 입력')
image = cv2.imread(r'이미지 경로 입력')
image_with_landmarks = image.copy()

# MediaPipe Pose를 이용하여 인스턴스를 생성합니다.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 이미지를 RGB로 변환합니다.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지에서 포즈를 분석합니다.
results = pose.process(image_rgb)

# 결과를 확인하고 각 부위의 좌표를 저장합니다.
if results.pose_landmarks is not None:
    landmarks = results.pose_landmarks.landmark
    landmarks_dict = {}
    for idx, landmark in enumerate(landmarks):
        landmarks_dict[idx] = [landmark.x * image_width_px, landmark.y * image_height_px]  # 이미지 크기에 맞게 좌표 조정

    # 길이 계산 함수 정의
    def calculate_length(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2)) / px_per_cm

    # 팔의 길이 계산
    right_arm_length = calculate_length(landmarks_dict[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks_dict[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_arm_length = calculate_length(landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks_dict[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # 다리의 길이 계산
    right_leg_length = calculate_length(landmarks_dict[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks_dict[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    left_leg_length = calculate_length(landmarks_dict[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks_dict[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # 어깨 너비 계산
    shoulder_width = calculate_length(landmarks_dict[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

    # 결과 출력
    print("Right Arm Length:", right_arm_length, "cm")
    print("Left Arm Length:", left_arm_length, "cm")
    print("Right Leg Length:", right_leg_length, "cm")
    print("Left Leg Length:", left_leg_length, "cm")
    print("Shoulder Width:", shoulder_width, "cm")

    # 이미지에 각 부위를 연결하는 선 그리기
    for landmark_id in mp_pose.PoseLandmark:
        landmark = landmarks[landmark_id]
        if landmark.visibility > 0.5:
            cv2.circle(image_with_landmarks, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 5, (255, 0, 0), -1)

    # 길이 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_color = (0, 0, 0)  # 검정색
    line_color = (0, 255, 0)
    line_thickness = 1
    cv2.putText(image_with_landmarks, f"Right Arm Length: {right_arm_length:.2f} cm", (50, 50), font, font_scale, font_color, line_thickness)
    cv2.putText(image_with_landmarks, f"Left Arm Length: {left_arm_length:.2f} cm", (50, 80), font, font_scale, font_color, line_thickness)
    cv2.putText(image_with_landmarks, f"Right Leg Length: {right_leg_length:.2f} cm", (50, 110), font, font_scale, font_color, line_thickness)
    cv2.putText(image_with_landmarks, f"Left Leg Length: {left_leg_length:.2f} cm", (50, 140), font, font_scale, font_color, line_thickness)
    cv2.putText(image_with_landmarks, f"Shoulder Width: {shoulder_width:.2f} cm", (50, 170), font, font_scale, font_color, line_thickness)

    cv2.imshow('Pose Detection', image_with_landmarks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
