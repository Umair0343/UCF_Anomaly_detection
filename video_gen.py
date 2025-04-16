import cv2
import numpy as np
import torch
from torchvision import transforms
from ucf import BBN  # Make sure this import is correct and available

def load_claws_model(model_path):
    input_size = 1024
    hidden_size = 512
    output_size = 32
    dropout_rate = 0.6
    model = BBN(input_size, hidden_size, output_size, dropout_rate)
    state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def process_with_npy_and_annotate_video(npy_path, video_path, model_path, output_path='output.mp4'):
    # Load model
    model = load_claws_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Load I3D features
    features = np.load(npy_path)  # Shape: (num_segments, 1024)
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # Get anomaly scores
    with torch.no_grad():
        scores = model(features_tensor)
        if scores.ndim > 1:
            scores = scores.mean(dim=1)  # (num_segments,)
        scores = torch.sigmoid(scores)  # Optional: make sure in [0,1]
    
    scores = scores.cpu().numpy()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Generate frame-to-score mapping
    num_segments = len(scores)
    segment_score_per_frame = np.interp(
        np.arange(total_frames),
        np.linspace(0, total_frames, num_segments),
        scores
    )

    # Read and annotate each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        score = segment_score_per_frame[frame_count]

        # Annotations
        font_scale = 0.4 
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (102, 255, 255)
        score_color = (0, 255, 255)

        cv2.putText(frame, f"Frame: {frame_count}", (10, 20), 
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Anomaly Score: {score:.4f}", (10, 40), 
                    font, font_scale, score_color, font_thickness, cv2.LINE_AA)

        # Score bar
        bar_length = 120
        bar_height = 8
        bar_x = 10
        bar_y = 50

        cv2.rectangle(frame, (bar_x, bar_y), 
                      (bar_x + int(score * bar_length), bar_y + bar_height), 
                      (0, 0, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), 
                      (bar_x + bar_length, bar_y + bar_height), 
                      (255, 255, 255), 1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Annotated video saved at: {output_path}")


# Example usage:
process_with_npy_and_annotate_video(
    npy_path='/home/iml1/Desktop/UMAIR/ucf/i3d_features/test/Robbery/Robbery137_x264.npy',
    video_path='/home/iml1/Desktop/UMAIR/ucf/Robbery137_x264.mp4',
    model_path='/home/iml1/Desktop/UMAIR/ucf/UCF_Models/best_clustering.pth',
    output_path='/home/iml1/Desktop/UMAIR/ucf/RoadAccidents022.mp4',
)
