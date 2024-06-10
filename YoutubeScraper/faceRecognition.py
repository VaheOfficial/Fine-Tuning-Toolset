import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import torch

# Initialize face detection and recognition models
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

def extract_faces(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    faces = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]
                faces.append(face)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    cap.release()
    return faces, timestamps

def recognize_faces(faces):
    embeddings = []
    for face in faces:
        face = Image.fromarray(face)
        face = mtcnn(face)
        if face is not None:
            face = face.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            embedding = resnet(face)
            embeddings.append(embedding.detach().cpu().numpy())
    return np.array(embeddings)

def match_faces_with_audio(diarization_result, face_timestamps, face_embeddings):
    # This function matches faces to speakers based on timestamps and embeddings
    speaker_face_map = {}
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        mid_turn = (turn.start + turn.end) / 2
        closest_face_idx = np.argmin([abs(ts - mid_turn) for ts in face_timestamps])
        speaker_face_map[speaker] = face_embeddings[closest_face_idx]
    return speaker_face_map
