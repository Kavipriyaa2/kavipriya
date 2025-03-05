import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
import threading
import queue as q  # Rename the imported module to avoid conflict
import pyperclip  # For copying to clipboard
import seaborn as sns
import matplotlib.pyplot as plt



# Load pre-trained face detection and recognition models
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the reference image (the image of the person to be recognized)
REFERENCE_IMAGE_PATH = r'C:\Users\rliki\OneDrive\Pictures\Camera Roll\mer.jpg'  # Replace with the path to your reference image

# Functions for encryption and decryption
def encrypt_message(message):
    ascii_values = [ord(char) for char in message]
    matrix_size = int(np.ceil(np.sqrt(len(ascii_values))))
    padded_values = ascii_values + [0] * (matrix_size ** 2 - len(ascii_values))
    matrix = np.array(padded_values).reshape(matrix_size, matrix_size)
    encrypted_matrix = matrix * 2  # Encryption logic (Multiply by 2)
    return encrypted_matrix

def decrypt_message(matrix):
    decrypted_matrix = matrix // 2  # Reverse operation (Divide by 2)
    ascii_values = decrypted_matrix.flatten()
    message = ''.join([chr(value) for value in ascii_values if value != 0])
    return message

# Function to Generate Heatmap
def show_heatmap(matrix, title="Encryption Heatmap"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(title)
    plt.show(block=False)  # Prevents Tkinter from freezing

# Function to train the face recognizer with the reference image
def train_face_recognizer():
    reference_image = cv2.imread(REFERENCE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        messagebox.showerror("Error", "Reference image not found.")
        exit()

    # Detect face in the reference image
    faces = face_detector.detectMultiScale(reference_image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        messagebox.showerror("Error", "No face detected in the reference image.")
        exit()

    (x, y, w, h) = faces[0]
    face_roi = reference_image[y:y + h, x:x + w]

    # Train the recognizer with the reference face
    face_recognizer.train([face_roi], np.array([1]))

# Function to recognize face from the webcam
def recognize_face():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # For each detected face, recognize it
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            # Predict the face
            label, confidence = face_recognizer.predict(face_roi)

            # If the confidence is low, it's a match
            if confidence < 50:  # Adjust this threshold as needed
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Match", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Facial Recognition", frame)
                cv2.waitKey(1000)  # Wait for 1 second to show the match
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Facial Recognition", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

# Function to handle encryption
def encrypt():
    message = simpledialog.askstring("Encrypt", "Enter the message to encrypt:")
    if message:
        encrypted_matrix = encrypt_message(message)
        encrypted_str = np.array2string(encrypted_matrix, separator=",", formatter={'all': lambda x: str(int(x))})
        pyperclip.copy(encrypted_str.replace("\n", "").replace(" ", ""))  # Copy encrypted matrix to clipboard
        messagebox.showinfo("Encrypted Matrix", f"Encrypted Matrix: {encrypted_str}\n(Copied to clipboard)")
        show_heatmap(encrypted_matrix, "Encryption Heatmap")

# Function to handle decryption
def decrypt():
    matrix_str = simpledialog.askstring("Decrypt", "Enter the encrypted matrix (comma-separated numbers):")
    try:
        matrix_str = matrix_str.replace("[", "").replace("]", "").replace("\n", "").replace(" ", "")
        matrix_values = list(map(int, matrix_str.split(',')))
        matrix_size = int(np.sqrt(len(matrix_values)))
        if matrix_size * matrix_size != len(matrix_values):
            raise ValueError("Matrix dimensions are not square!")
        matrix = np.array(matrix_values).reshape(matrix_size, matrix_size)
        decrypted_message = decrypt_message(matrix)
        messagebox.showinfo("Decrypted Message", f"Decrypted Message: {decrypted_message}")
    except Exception as e:
        messagebox.showerror("Error", f"Error decrypting the message: {e}")

# Function to start face recognition in a separate thread
def start_face_recognition(queue):
    if recognize_face():
        queue.put("SUCCESS")  # Notify the main thread that face recognition succeeded
    else:
        queue.put("FAILURE")  # Notify the main thread that face recognition failed

# Main program
def main():
    # Train the face recognizer with the reference image
    train_face_recognizer()

    # Create the Tkinter root window
    global root
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create a queue for communication between threads
    comm_queue = q.Queue()  # Use a different variable name to avoid conflict

    # Start face recognition in a separate thread
    threading.Thread(target=start_face_recognition, args=(comm_queue,)).start()

    # Check the queue for messages from the background thread
    def check_queue():
        try:
            message = comm_queue.get_nowait()
            if message == "SUCCESS":
                # Create a new window for encryption/decryption options
                option_window = tk.Toplevel(root)
                option_window.title("Choose Option")
                option_window.geometry("300x150")

                # Add buttons for encryption and decryption
                encrypt_button = tk.Button(option_window, text="Encrypt Message", command=encrypt, width=20, height=2)
                encrypt_button.pack(pady=10)

                decrypt_button = tk.Button(option_window, text="Decrypt Message", command=decrypt, width=20, height=2)
                decrypt_button.pack(pady=10)

                # Close the main loop when the option window is closed
                option_window.protocol("WM_DELETE_WINDOW", root.quit)
            else:
                messagebox.showerror("Error", "No matching face detected. Exiting...")
                root.quit()  # Close the Tkinter main loop
        except q.Empty:
            root.after(100, check_queue)  # Check the queue again after 100ms

    # Start checking the queue
    root.after(100, check_queue)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
