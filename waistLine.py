import cv2
import mediapipe as mp
import numpy as np

# MediaPipe와 OpenCV 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 이미지 로드
image_path = "C:/image/111.jpg"  # 이미지 경로 설정
image = cv2.imread(image_path)

image_height, image_width, _ = image.shape

# Pose estimation
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        raise ValueError("Pose landmarks could not be detected.")

 
    # 1-1. 왼쪽 어깨 위치 추정
    left_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y])

    # 1-2. 왼쪽 엉덩이 위치 추정
    left_hip = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, 
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])

    # 1-3. 왼쪽 어깨와 왼쪽 엉덩이 사이의 중간점 계산
    left_waist_position = (left_shoulder + left_hip) / 2
    left_waist_position_pixel = np.round(left_waist_position * [image_width, image_height]).astype(int)


    # 결과 표시: 왼쪽 허리 위치에 점 그리기 (빨간색)
    cv2.circle(image, tuple(left_waist_position_pixel), 5, (0, 0, 255), -1)

    # 오른쪽 어깨 위치 추정
    right_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])

    # 오른쪽 엉덩이 위치 추정
    right_hip = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y])

    # 오른쪽 어깨와 오른쪽 엉덩이 사이의 중간점 계산
    right_waist_position = (right_shoulder + right_hip) / 2
    right_waist_position_pixel = np.round(right_waist_position * [image_width, image_height]).astype(int)

    # 결과 표시: 오른쪽 허리 위치에 점 그리기 (빨간색)
    cv2.circle(image, tuple(right_waist_position_pixel), 5, (0, 0, 255), -1)

    # 허리 위치를 선으로 연결
    cv2.line(image, tuple(left_waist_position_pixel), tuple(right_waist_position_pixel), (255, 0, 0), 2)

    # 결과 표시
    cv2.imshow('Waist Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

     # 1-1. 정수리 위치 추정
    left_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y])

    # 1-2. 발바닥 위치 추정
    left_hip = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, 
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])

    # 1-3. 왼쪽 어깨와 왼쪽 엉덩이 사이의 중간점 계산
    left_waist_position = (left_shoulder + left_hip) / 2
    left_waist_position_pixel = np.round(left_waist_position * [image_width, image_height]).astype(int)


    # 결과 표시: 왼쪽 허리 위치에 점 그리기 (빨간색)
    cv2.circle(image, tuple(left_waist_position_pixel), 5, (0, 0, 255), -1)

    # 오른쪽 어깨 위치 추정
    right_shoulder = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])

    # 오른쪽 엉덩이 위치 추정
    right_hip = np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, 
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y])

    # 오른쪽 어깨와 오른쪽 엉덩이 사이의 중간점 계산
    right_waist_position = (right_shoulder + right_hip) / 2
    right_waist_position_pixel = np.round(right_waist_position * [image_width, image_height]).astype(int)

    # 결과 표시: 오른쪽 허리 위치에 점 그리기 (빨간색)
    cv2.circle(image, tuple(right_waist_position_pixel), 5, (0, 0, 255), -1)

    # 허리 위치를 선으로 연결
    cv2.line(image, tuple(left_waist_position_pixel), tuple(right_waist_position_pixel), (255, 0, 0), 2)

    # 결과 표시
    cv2.imshow('Waist Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        
    


        import cv2

        # 이미지 로드
        image_path = "C:/image/111.jpg"
        image = cv2.imread(image_path)

        if image is None:
            print("Image not loaded. Check the file path.")
        else:
            # 폰트 설정
            font = cv2.FONT_HERSHEY_PLAIN
    
        # 텍스트 색상 정의 (B, G, R) - 파란색
            blue = (255, 0, 0)
    
        # 이미지에 텍스트 추가
            text = "Estimated Waist: 75.65 cm"
            cv2.putText(image, text, (350, 40), font, 2, blue, 1, cv2.LINE_AA)
    
        # 이미지 표시
            cv2.imshow('Image with Estimated Waist Circumference', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

   

