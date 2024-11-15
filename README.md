
# SILICAI: Semiconductor Defect Analysis Chatbot

SILICAI is a full-stack application designed for semiconductor defect analysis. It combines an intelligent chatbot with image classification and segmentation features for wafer defect detection, powered by deep learning models. Users can interact with the bot to get insights about semiconductor defects and upload wafer images for classification or segmentation.

## Features

- **Chatbot Interaction**: Communicate with the bot for general knowledge about semiconductor defects.
- **Classify Defects**: Upload a wafer image to classify the defect type (e.g., Loc, Scratch, Edge-Ring).
- **Segment Wafer Map**: Upload a wafer map image to perform segmentation and overlay the results.
- **Memory Purge**: Clear the conversation history if needed.
- **Authentication**: User login and registration with JWT token support.

## Setup Instructions

### Prerequisites

1. Install Python (>=3.7) and Node.js (>=14).
2. Install dependencies:

   **Backend:**
   - Install Python dependencies using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

   **Frontend:**
   - Navigate to the frontend folder and install Node.js dependencies:
     ```bash
     cd frontend
     npm install
     ```

### Running the Application

1. **Start Backend (FastAPI) Server:**
   In the project root, run the following command:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   This starts the FastAPI server, which hosts the APIs for the chatbot, classification, and segmentation tasks.

2. **Start Frontend (React) App:**
   Navigate to the frontend directory and start the React development server:
   ```bash
   npm start
   ```

   This opens the app in the browser at `http://localhost:3000`.

### API Endpoints

#### User Authentication

- **POST /register**: Register a new user with username and password.
- **POST /token**: Login to get a JWT token for authentication.
- **GET /protected-route**: Access protected routes using the JWT token.

#### Chatbot

- **POST /chat**: Ask the bot a question about semiconductor defects. The bot provides responses based on previous conversation context.

#### Image Classification and Segmentation

- **POST /ml**: Classify a wafer map image and return the defect type.
- **POST /dl**: Segment a wafer map image and return the segmented image.

#### Conversation History

- **GET /history**: Retrieve conversation history. Optionally purge it with `purge=True`.

### File Upload

Users can upload images in the classification or segmentation sections. The app supports image formats such as JPG, PNG, and JPEG.

### Frontend

The frontend is built with React and provides a user-friendly interface for interacting with the chatbot, uploading images, and viewing classification or segmentation results.

## Dependencies

### Backend
- FastAPI
- PyTorch
- Langchain
- SQLite3
- Uvicorn

### Frontend
- React
- React-toastify
- Marked (for markdown rendering)