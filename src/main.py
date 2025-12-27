import sys
import os

# Check if cv2 is installed
try:
    import cv2  # Add this import to check for cv2.dnn
except ImportError:
    print("Error: OpenCV is not installed. Please install it using 'pip install opencv-python'.")
    sys.exit(1)

# Check if cv2 has dnn attribute
if not hasattr(cv2, 'dnn'):
    print("Error: Your OpenCV installation does not have the 'dnn' module.")
    sys.exit(1)

from pretrained_static import detect_age_gender
from pretrained_realtime import detect_age_gender_video


def main():
    # Default paths (relative to src/)
    default_image = os.path.join("D:\Age and Gender Identification\data\test_images", "person1.jpg")
    
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py [static|realtime|train|custom] [image_path (optional for static/custom)]")
        print("Example: python main.py static ../data/test_images/person1.jpg")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    image_paths = sys.argv[2:] if len(sys.argv) > 2 else [default_image]

    # Execute based on mode
    try:
        if mode == "static":
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"Error: Image {image_path} not found. Skipping...")
                    continue
                print(f"Running static prediction on {image_path}")
                detect_age_gender(image_path)
                print(f"Completed prediction for {image_path}")
        
        elif mode == "realtime":
            print("Running real-time video detection (press 'q' to quit)")
            detect_age_gender_video()
        
        elif mode == "train":
            print("Training custom model (this may take a while)")
            
        
        elif mode == "custom":
            if not os.path.exists(image_paths[0]):
                print(f"Error: Image {image_paths[0]} not found. Using default if available.")
                image_paths[0] = default_image
            print(f"Running custom model prediction on {image_paths[0]}")
            
        
        else:
            print(f"Error: Unknown mode '{mode}'. Use: static, realtime, train, or custom.")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()