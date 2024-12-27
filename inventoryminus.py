import cv2
import pymongo
from pymongo import MongoClient
import datetime
import time
import torch
import pygame

pygame.mixer.init()

# MongoDB Atlas Connection String
MONGO_URI ='mongodb+srv://ramankattisubhash7:ramankattisubhash7@cluster0.3vcdp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
DATABASE_NAME = "Inventory"
ITEMS_COLLECTION_NAME = "items"
BILLS_COLLECTION_NAME = "bill"
ALLOWED_CLASSES = ['apple', 'banana', 'bottle', 'knife','toothbrush','book','orange' ]

# Load YOLOv5 Model
def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Or load from local path if you have downloaded it
    return model

# Connect to MongoDB
def connect_to_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    items_collection = db[ITEMS_COLLECTION_NAME]
    bills_collection = db[BILLS_COLLECTION_NAME]
    return items_collection, bills_collection

# Function to update stock in the 'items' collection (deducting quantity)
def update_item_stock(item_name, items_collection):
    # Find the item in the database by name
    item = items_collection.find_one(
        {"itemName": {'$regex': f'^{item_name}$', '$options': 'i'}}  # Case-insensitive match
    )

    if item:
        if item['stock'] > 0:
            # Decrease stock by 1 (deduct the quantity)
            new_stock = item['stock'] - 1
            # Update the item stock in the database
            items_collection.update_one(
                {"_id": item["_id"]},
                {"$set": {"stock": new_stock}}
            )
            print(f"Updated stock for '{item_name}' to {new_stock}.")
        else:
            print(f"Item '{item_name}' is out of stock.")
    else:
        print(f"Item '{item_name}' not found in the database. Please update the item database.")

# Process each frame from the camera feed
def process_frame(frame, model, items_collection, basket_id, basket_no, detected_items):
    results = model(frame)  # Perform inference
    detections = results.pandas().xywh[0]  # Get detection results in pandas format
    detections = detections[detections['confidence'] > 0.3]  # Lower confidence threshold
    detections = detections[detections['name'].isin(ALLOWED_CLASSES)]

    if len(detections) > 0:
        for _, row in detections.iterrows():
            detected_item_name = row['name']
            print(f"Item detected: {detected_item_name}")  # Print detected item

            # Check if the item exists in the database
            item_in_db = items_collection.find_one(
                {"itemName": {'$regex': f'^{detected_item_name}$', '$options': 'i'}}  # Case-insensitive match
            )

            if item_in_db:
                # Update the stock of the detected item (deduct stock)
                update_item_stock(detected_item_name, items_collection)
                return True  # Item successfully processed
            else:
                print(f"Item '{detected_item_name}' is not in the database. Please update the database.")
                return False  # Continue processing frames

    return False  # No valid items detected

# Main function to capture frames from the camera
def main(basket_id, basket_no):
    model = load_yolov5_model()  # Load YOLOv5 model
    items_collection, bills_collection = connect_to_mongo()  # Connect to MongoDB
    detected_items = {}  # Dictionary to track detected items

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Starting automatic detection...")
    item_detected = False  # Flag to stop finding once the first item is detected

    while not item_detected:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Automatically process the frame
        item_detected = process_frame(frame, model, items_collection, basket_id, basket_no, detected_items)

    pygame.mixer.music.load("stock.mp3")  # Replace with the path to your MP3 file
    pygame.mixer.music.play()

    # Wait until the music finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    print("Detection process complete. Stopping...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    basket_id = "12345"
    basket_no = "BASKET001"
    main(basket_id, basket_no)
