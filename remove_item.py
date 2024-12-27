import torch
import cv2
from pymongo import MongoClient
import pygame

from Bill import calculate_grand_total

pygame.mixer.init()
# MongoDB Atlas Connection String
MONGO_URI = "mongodb+srv://sachinb665:sachinb665@cluster0.ofadrht.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "bill"
ITEMS_COLLECTION_NAME = "items"
BILLS_COLLECTION_NAME = "bill"

ALLOWED_CLASSES = ['apple', 'banana', 'orange', 'bottle', 'knife', 'chair']
# Load YOLOv5 Model
def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model (can be adjusted)
    return model


# Connect to MongoDB
def connect_to_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    items_collection = db[ITEMS_COLLECTION_NAME]
    bills_collection = db[BILLS_COLLECTION_NAME]
    return items_collection, bills_collection


# Function to decrease the quantity of an item in the bill
def decrease_item_quantity(basket_id, item_name):
    items_collection, bills_collection = connect_to_mongo()

    # Find the bill entry using the provided basket_id
    bill = bills_collection.find_one({"basketId": basket_id, "paymentStatus": "Unpaid"})

    if bill:
        # Find the item in the bill by name (case-insensitive match)
        item_in_bill = next((item for item in bill["items"] if item["itemName"].lower() == item_name.lower()), None)

        if item_in_bill:
            # If item exists and quantity is more than 1, decrease its quantity by 1
            if item_in_bill["quantity"] >= 1:
                new_quantity = item_in_bill["quantity"] - 1
                new_item_total = item_in_bill["itemPrice"] * new_quantity

                # Calculate GST and item discount from the item collection
                item_data = items_collection.find_one({"name": {'$regex': f'^{item_name}$', '$options': 'i'}})
                if item_data:
                    gst = item_data.get("gst", 0.18)  # Get GST, default is 18%
                    discount = item_data.get("discount", 0)  # Get item discount, default is 0%
                else:
                    gst = 0.18  # Default GST value
                    discount = 0  # Default discount value

                # Apply GST and discount to the new total
                gst_amt = float(gst) * new_item_total
                discounted_total = new_item_total - discount
                total = discounted_total + gst_amt

                # Update the item quantity, totals, and GST in the bill
                bills_collection.update_one(
                    {"basketId": basket_id, "paymentStatus": "Unpaid", "items.itemName": item_name},
                    {"$set": {
                        "items.$.quantity": new_quantity,
                        "items.$.itemTotal": new_item_total,
                        "items.$.gstAmt": gst_amt,
                        "items.$.total": total
                    }}
                )
                print(f"Decreased quantity for '{item_name}' to {new_quantity} in the bill with basket ID: {basket_id}")
                pygame.mixer.music.load("removed.mp3")  # Replace with the path to your MP3 file
                pygame.mixer.music.play()

                # Wait until the music finishes playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

            else:
                print(
                    f"Item '{item_name}' has only one quantity in the bill with basket ID {basket_id}, cannot decrease further.")

        else:
            print(f"Item '{item_name}' not found in the bill with basket ID {basket_id}.")

        # Recalculate the grand total of the bill (sum all item totals)
        updated_bill = bills_collection.find_one({"basketId": basket_id, "paymentStatus": "Unpaid"})
        grand_total = calculate_grand_total(updated_bill['items'])
        # Update the grand total for the bill
        bills_collection.update_one(
            {"basketId": basket_id, "paymentStatus": "Unpaid"},
            {"$set": {"grandTotal": grand_total}}
        )
        print(f"Updated grand total for basket ID {basket_id}: {grand_total}")
    else:
        print(f"Bill with basket ID {basket_id} not found or already paid.")


# Function to process the frame and detect items
def process_frame(frame, model, basket_id):
    results = model(frame)  # Perform inference with YOLOv5
    detections = results.pandas().xywh[0]  # Get detections in pandas format
    detections = detections[detections['confidence'] > 0.3]  # Filter low-confidence detections
    detections = detections[detections['name'].isin(ALLOWED_CLASSES)]
    # If any item is detected, return the first one (or customize to your needs)
    if not detections.empty:
        detected_item_name = detections.iloc[0]['name'].lower()  # Get the first detected item name
        print(f"Detected item: {detected_item_name}")
        return detected_item_name
    else:
        print("No items detected.")
        return None


# Main function to detect and update item quantity
def detect_and_update_item_quantity(basket_id):
    model = load_yolov5_model()  # Load YOLOv5 model
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect item in the frame
        detected_item_name = process_frame(frame, model, basket_id)

        if detected_item_name:
            # Decrease the quantity of the detected item in the bill
            decrease_item_quantity(basket_id, detected_item_name)
            break  # Exit after detecting and updating the item


        # Display the current frame
        cv2.imshow('YOLOv5 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()


# If you want to run this function with a specific basket ID
if __name__ == "__main__":
    basket_id = "12345"  # Replace with the actual basket ID
    detect_and_update_item_quantity(basket_id)
