import cv2
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

# Load the pre-trained model and processor from Hugging Face
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Set up the device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Open the laptop camera using OpenCV
cap = cv2.VideoCapture(0)  # 0 is typically the default laptop camera

# Define COCO categories (common object categories)
COCO_LABELS = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", 
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", 
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "toilet", "N/A", 
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

while cap.isOpened():
    ret, frame = cap.read()  # Read frame from the camera
    if not ret:
        break

    # Convert the frame to RGB (required by the model)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare the image for the model
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)

    # Perform inference (object detection)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract bounding boxes and labels
    target_sizes = torch.tensor([rgb_frame.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Draw bounding boxes on the frame
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.7:  # Show only predictions with a confidence score > 0.7
            box = [int(i) for i in box.tolist()]  # Convert float to integer
            category = COCO_LABELS[label]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"{category} ({score:.2f})", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the video with bounding boxes
    cv2.imshow('Hugging Face Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()