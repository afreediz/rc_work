import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import pickle
import re
import vecvalidate as validate
import datamodel as datamodel 
import boto3
import os
from dotenv import load_dotenv
load_dotenv()

class LicensePlateDetector:
    def __init__(self, yolo_model_path):
        # Load YOLO model from pickle
        with open(yolo_model_path, 'rb') as f:
            self.yolo_model = pickle.load(f)
        
        self.validator = validate.VehicleNumberValidator()
        self.status = "stage0"
        self.userid = 100
        self.db = datamodel.VehicleDatabase("vehicles.db")
        
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        return opening
    
    def init_s3(self, access_key, secret_key):
        self.s3 = boto3.client('s3',
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key)
        
    def stream_video_from_s3(self, bucket_name, key):
        url = self.s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': key})
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Error opening video stream from {url}")
            return None
        return cap
    
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Inverse binary threshold
        return thresh

    def rotate_frame(self, frame):
        # Rotate the frame 90 degrees clockwise
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated_frame

    def detect_and_extract(self, bucket_name, key):

        if not self.s3:
            print("Please initalize the s3 bucket.")
            
        # filename = os.path.basename(key)
        # self.s3.download_file(bucket_name, key, "eg.mp4")
        # print(filename)
        # cap = cv2.VideoCapture("bli.mp4")

        cap = self.stream_video_from_s3(bucket_name, key)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FPS: {fps}, Frame Count: {frame_count}, Width: {width}, Height: {height}")

        # while the video is opened
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.rotate_frame(frame)
            # if frame is None:
            #     print("Error: Unable to load image from:", image_path)
            #     return

            # Make predictions with the YOLO model
            results = self.yolo_model(frame)

            if len(results) == 0:
                print("Error: No objects detected in the image.")
                return

            # Iterate over the detection outputs
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()

                for i, box in enumerate(boxes):
                    # Assuming car label is 1 and truck label is 2 (modify according to your model)
                    if classes[i] in [0] and scores[i] > 0.6:  # 1 and 2 are the classes for cars and trucks
                        # Get the bounding box coordinates
                        x1, y1, x2, y2 = [int(coord) for coord in box]

                        # Crop the detected vehicle
                        vehicle = frame[y1:y2, x1:x2]
                        vehicle_gray = self.preprocess_image(vehicle)

                        cv2.imwrite(f"dbg/debug_vehicle_{i}.png", vehicle_gray)

                        # Use pytesseract to extract the text (license plate)
                        text = pytesseract.image_to_string(vehicle_gray, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 11')

                        # Extract the text from the result
                        cleaned_license_plate = re.sub(r'\s|[^A-Za-z0-9]', '', text)
                        print("result ", cleaned_license_plate)
                        if self.validator.validate_vehicle_number(cleaned_license_plate):
                            print(f"{cleaned_license_plate} is a valid Indian vehicle number.")
                            self.db.insert_vehicle(vehicle_number=cleaned_license_plate, owner_name="", vehicle_class="", contact_number="", html_page="", source="", image="", status=self.status,user_id=self.userid)
                        else:
                            print(f"{cleaned_license_plate} is not a valid Indian vehicle number.")

                        # Draw the bounding box and text on the frame
                        if text:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the processed image to the output file
        # cv2.imwrite(output_path, frame)

# Example usage
detector = LicensePlateDetector('best.pickle')
detector.init_s3(os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
detector.detect_and_extract("rc-bucket-1", "videos/REC1730047643518045176.mp4")