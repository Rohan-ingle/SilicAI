import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import shutil  # For saving the uploaded image temporarily
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import asyncio
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
import json
import os
import warnings
from langchain.schema import Document, HumanMessage

import fitz

import sqlite3
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

from typing import List

# Constants for JWT
SECRET_KEY = "keysec"  # Replace with a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI()
# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database setup function
def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the database with a users table
def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

# Call to initialize the database
init_db()

# Pydantic models for request and response
class User(BaseModel):
    username: str
    password: str


# Database model for the user response
class UserData(BaseModel):
    id: int
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Helper function to hash passwords
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Helper function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# JWT token creation
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Registration endpoint
@app.post("/register", response_model=dict)
async def register(user: User):
    hashed_password = hash_password(user.password)
    try:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user.username, hashed_password))
            conn.commit()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")

# Login endpoint to authenticate and return JWT
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (form_data.username,))
        user = cursor.fetchone()

    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Dependency to get the current user from the token
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
    if user is None:
        raise credentials_exception
    return user

# Protected route to test authentication
@app.get("/protected-route")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello, {current_user['username']}! This is a protected route."}


# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variable for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


@app.get("/users", response_model=List[UserData])
async def get_all_users():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username FROM users")
            users = cursor.fetchall()
        
        user_list = [{"id": user["id"], "username": user["username"]} for user in users]
        return user_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load the JSON file and prepare the documents
local_path = "context.json"  # Path to your JSON dataset
json_data = load_json(local_path)

# Prepare documents by extracting the content from JSON
documents = []
for chunk in json_data:
    # Check if content is a list (for FAQ sections or other list-based content)
    if isinstance(chunk["content"], list):
        # Convert the list to a string (e.g., joining questions and answers)
        content_str = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in chunk["content"]])
    else:
        content_str = chunk["content"]
    
    # Append the content as a Document
    documents.append(Document(page_content=content_str))

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)

# Load language model
local_model = "llama3.2:1b"
llm = ChatOllama(model=local_model)

# Define the smart prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are SilicAI, a QnA Chatbot that helps users with semiconductor defect analysis and general knowledge.
- Only greet of introduce if asked
- Do not mention the context unless it's directly relevant to the question.
- Do not mention conversation history from context at all
- In context if there is ignore context, ignore the part of the context before the ignore context mention.
- Don't answer anything that is not relevant to the question
- If the user greets you or asks personal questions (e.g., "hello", "hi", "who are you"), respond with a polite greeting or a brief introduction about yourself.
- If told to ignore context, only answer the latest question without considering previous interactions and context.
- If told to forget context, only answer the latest question without considering previous interactions and context.
- when told to consider context, consider last few interactions and context to provide a coherent response.
- If the user asks a technical or document-related question and the context from the PDF document contains relevant information, provide a concise answer based on that context.
- If the context doesn't address the question or is not needed, provide a brief and relevant answer using your general knowledge.
- Do not mention the context unless it's directly relevant to the question.
- Maintaining Conversation Flow:
- Reference previous interactions if they are relevant to the current question to provide coherent and connected responses.
- Avoid resetting or forgetting the context unless the user explicitly requests to start a new topic.
- General Knowledge Questions: If the context does not contain relevant information for the user's question, respond using your general knowledge without referencing the context.


Question: {question}
Context: {context}
Answer:""",
)



# Set up retriever
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# Define conversation history storage
conversation_history = []

# Pydantic model to receive input as JSON
class ChatRequest(BaseModel):
    question: str
    defect: str = ""

# Process tokens for streaming response (only output the token content)
async def process_tokens(stream, final_answer_list):
    queue = Queue()

    def run_stream():
        try:
            for token in stream:
                # Extract content if it exists, ignoring other metadata
                token_text = getattr(token, 'content', None)
                if token_text:
                    final_answer_list.append(token_text)  # Collect answer for conversation history
                    queue.put_nowait(json.dumps({"token": token_text}) + "\n")
        except Exception as e:
            queue.put_nowait(json.dumps({"error": str(e)}) + "\n")
        finally:
            queue.put_nowait(None)

    # Run the synchronous stream in a background thread
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, run_stream)

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item.encode('utf-8')

# Global variable to store the current defect type from classification
current_defect_type = None

# Define the /chat endpoint with updated context to include current_defect_type
@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """
    Endpoint for chatting with the PDF.
    Accepts a question and defect information, and returns an answer from the PDF.
    """
    formatted_question = request.question.strip()
    global conversation_history, current_defect_type

    # Format conversation history for context
    context = "\n".join([
        f"User: {exchange['question']}\nBot: {exchange['answer']}"
        for exchange in conversation_history
    ])

    # Add the latest question to the context
    context += f"\nUser: {formatted_question}"

    # Include current defect type in the context if it is set
    if current_defect_type is not None:
        context += f"\nIgnore any defect mentioned before in the context do not mention it at all, now an image of wafer map was uploaded and Current Defect Type is {current_defect_type}."

    # Format prompt with the conversation context
    prompt_text = QUERY_PROMPT.format(question=formatted_question, context=context)
    message = [HumanMessage(content=prompt_text)]

    try:
        # Stream the response from the language model
        stream = llm.stream(message)

        # Initialize a list to capture the final answer text for storage
        final_answer_list = []

        # Return streaming response to the client
        response = StreamingResponse(process_tokens(stream, final_answer_list), media_type="application/json")

        # Append current interaction to conversation history after streaming completes
        conversation_history.append({"question": formatted_question, "answer": "".join(final_answer_list)})
        
        return response
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# New endpoint to retrieve and optionally purge the conversation history
@app.get("/history")
async def get_conversation_history(purge: bool = Query(False)):
    """
    Endpoint to retrieve the entire conversation history for debugging purposes.
    If 'purge' is set to true, it will clear the history after returning it.
    """
    global conversation_history, current_defect_type

    # Capture the current history and defect type to return
    history = {
        "conversation_history": conversation_history,
        "current_defect_type": current_defect_type
    }

    # Clear the history and defect type if purge is set to true
    if purge:
        conversation_history = []
        current_defect_type = None

    return JSONResponse(content=history)

# ----------------------- Classification Part -----------------------

# Define the Classification Model with 9 classes
class ClassificationModel(nn.Module):
    def __init__(self, num_classes=9):
        super(ClassificationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

# Load your classification model (replace with your actual model path)
classification_model = ClassificationModel(num_classes=9)
CLASSIFICATION_MODEL_PATH = "classifier_model.pth"
classification_model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=torch.device('cpu')))
classification_model.eval()

# If using GPU, move the model to CUDA
if torch.cuda.is_available():
    classification_model = classification_model.cuda()

# Reverse the intensity map
intensity_map_reverse = {0: 0, 128: 1, 255: 2}

# Function to convert image back to wafer map (numpy array)
def image_to_wafer_map(image_path):
    """
    Converts a grayscale image to a wafer map with intensities 0, 1, 2.
    
    Args:
        image_path (str): Path to the input grayscale image.
    
    Returns:
        np.ndarray: Wafer map with values 0, 1, 2.
    """
    wafer_image = Image.open(image_path).convert("L")  # Convert to grayscale
    wafer_map = np.array(wafer_image)
    
    # Map 0, 128, 255 to 0, 1, 2 respectively using thresholding
    wafer_map = np.where(wafer_map < 64, 0, np.where(wafer_map < 192, 1, 2))
    return wafer_map

# Function to classify the wafer map using the model
def classify_wafer_map(wafer_map_tensor, model):
    """
    Classifies the wafer map to identify the defect type.
    
    Args:
        wafer_map_tensor (torch.Tensor): Tensor representation of the wafer map.
        model (nn.Module): Classification model.
    
    Returns:
        int: Defect type index.
    """
    if torch.cuda.is_available():
        wafer_map_tensor = wafer_map_tensor.cuda()
    with torch.no_grad():
        output = model(wafer_map_tensor)
    _, predicted = torch.max(output, 1)
    defect_type = predicted.item()
    return defect_type

# Defect mapping
defect_mapping = {
    0: "no defect",
    1: "Loc defect",
    2: "Edge-Loc defect",
    3: "Center defect",
    4: "Edge-Ring defect",
    5: "Scratch defect",
    6: "Random defect",
    7: "Near-full defect",
    8: "Donut defect"
}

# Function to convert image to tensor for classification
def image_to_tensor_classification(image_path):
    """
    Converts a wafer map image to a PyTorch tensor.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        torch.Tensor: Tensor suitable for the classification model.
    """
    wafer_map = image_to_wafer_map(image_path)
    wafer_map_tensor = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    return wafer_map_tensor

# Function to classify the wafer map
def classify_wafer_map_function(wafer_map_tensor, model):
    """
    Classifies the wafer map to identify the defect type.
    
    Args:
        wafer_map_tensor (torch.Tensor): Tensor representation of the wafer map.
        model (nn.Module): Classification model.
    
    Returns:
        int: Defect type index.
    """
    if torch.cuda.is_available():
        wafer_map_tensor = wafer_map_tensor.cuda()
    with torch.no_grad():
        output = model(wafer_map_tensor)
    _, predicted = torch.max(output, 1)
    defect_type = predicted.item()
    return defect_type

# Function to convert wafer map to image (for visualization)
def wafer_map_to_image(wafer_map):
    """
    Converts a wafer map with values 0, 1, 2 to a grayscale image with intensities 0, 128, 255.
    
    Args:
        wafer_map (np.ndarray): Wafer map with values 0, 1, 2.
    
    Returns:
        PIL.Image.Image: Grayscale image.
    """
    intensity_map = {0: 0, 1: 128, 2: 255}
    wafer_map_visual = np.vectorize(intensity_map.get)(wafer_map)
    wafer_map_visual = wafer_map_visual.astype(np.uint8)
    wafer_map_image = Image.fromarray(wafer_map_visual, mode='L')
    return wafer_map_image

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        
        # Shared encoder (Convolutional layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Segmentation decoder (Transpose convolution layers for upsampling)
        self.segmentation_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Sigmoid for binary segmentation
        )

    def forward(self, x):
        encoded_features = self.encoder(x)
        segmentation_output = self.segmentation_decoder(encoded_features)
        return segmentation_output

# Initialize the Segmentation Model
segmentation_model = SegmentationModel()

# Load the state dictionary (replace 'segmentation_model.pth' with your actual model path)
SEGMENTATION_MODEL_PATH = "best_model.pth"
segmentation_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=torch.device('cpu')))
segmentation_model.eval()

# If using GPU, move the model to CUDA
if torch.cuda.is_available():
    segmentation_model = segmentation_model.cuda()


# Function to convert image to tensor for segmentation
def image_to_tensor(image_path):
    """
    Converts a wafer map image to a PyTorch tensor.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        torch.Tensor: Tensor suitable for the segmentation model.
    """
    wafer_map = image_to_wafer_map(image_path)
    wafer_map_tensor = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    return wafer_map_tensor

# Function to perform segmentation
def perform_segmentation(tensor, model):
    """
    Performs segmentation on the input tensor using the provided model.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        model (nn.Module): Segmentation model.
    
    Returns:
        np.ndarray: Segmented map.
    """
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        segmentation_output = model(tensor)
    segmented_map = segmentation_output.cpu().squeeze().numpy()
    return segmented_map

# Function to create overlay image
def create_overlay_image(original_image_path, segmented_map):
    """
    Creates an overlay image by combining the original wafer map and the segmented map.
    
    Args:
        original_image_path (str): Path to the original grayscale image.
        segmented_map (np.ndarray): Segmented map with float values between 0 and 1.
    
    Returns:
        str: Base64-encoded PNG image.
    """
    wafer_map = np.array(Image.open(original_image_path).convert("L"))  # Original grayscale image

    # Create a binary mask from the segmented map
    binary_mask = (segmented_map > 0.5).astype(np.uint8)  # Threshold at 0.5

    # Create an RGB image for overlay
    overlay_blue = np.zeros((*wafer_map.shape, 3), dtype=np.uint8)
    overlay_blue[..., 2] = binary_mask * 255  # Blue channel

    # Combine original image and overlay
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(wafer_map, cmap='gray')
    ax.imshow(overlay_blue, alpha=0.3)
    ax.axis("off")

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    overlay_image = base64.b64encode(buf.read()).decode('utf-8')
    return overlay_image

# Define the /dl endpoint for segmentation
@app.post("/dl")
async def segment_image(image: UploadFile = File(...)):
    """
    Endpoint for image segmentation.
    Accepts an uploaded wafer map image, performs segmentation, and returns the annotated image.
    
    Args:
        image (UploadFile): Uploaded image file.
    
    Returns:
        JSONResponse: Contains a message and the base64-encoded segmented image.
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            return JSONResponse(content={"error": "Invalid file type. Please upload an image."}, status_code=400)
        
        # Save the uploaded file temporarily
        temp_file_path = "temp_uploaded_image_segment.png"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Convert image to tensor
        wafer_map_tensor = image_to_tensor(temp_file_path)

        # Perform segmentation
        segmented_map = perform_segmentation(wafer_map_tensor, segmentation_model)

        # Create overlay image
        overlay_image_base64 = create_overlay_image(temp_file_path, segmented_map)

        # Remove the temporary file
        os.remove(temp_file_path)

        # Return the base64 image
        result = {
            "message": "I have segmented and annotated the input wafer map.",
            "segmented_image": overlay_image_base64
        }
        return result
    except Exception as e:
        # In case of any errors, ensure the temporary file is removed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ----------------------- Classification Part -----------------------

# Function to classify the wafer map
def classify_wafer_map_function(wafer_map_tensor, model):
    """
    Classifies the wafer map to identify the defect type.
    
    Args:
        wafer_map_tensor (torch.Tensor): Tensor representation of the wafer map.
        model (nn.Module): Classification model.
    
    Returns:
        int: Defect type index.
    """
    if torch.cuda.is_available():
        wafer_map_tensor = wafer_map_tensor.cuda()
    with torch.no_grad():
        output = model(wafer_map_tensor)
    _, predicted = torch.max(output, 1)
    defect_type = predicted.item()
    return defect_type

# Function to convert image to tensor for classification
def image_to_tensor_classification(image_path):
    """
    Converts a wafer map image to a PyTorch tensor for classification.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        torch.Tensor: Tensor suitable for the classification model.
    """
    wafer_map = image_to_wafer_map(image_path)
    wafer_map_tensor = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    return wafer_map_tensor

# Modify the /ml endpoint to update the current_defect_type
@app.post("/ml")
async def classify_image(image: UploadFile = File(...)):
    """
    Endpoint for image classification.
    Accepts an uploaded wafer map image, classifies it, and returns the defect type.
    
    Args:
        image (UploadFile): Uploaded image file.
    
    Returns:
        JSONResponse: Contains the defect type and name.
    """
    global current_defect_type  # Use the global variable to store the latest defect type

    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            return JSONResponse(content={"error": "Invalid file type. Please upload an image."}, status_code=400)
        
        # Save the uploaded file temporarily
        temp_file_path = "temp_uploaded_image_classify.png"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Convert image to tensor for classification
        wafer_map_tensor = image_to_tensor_classification(temp_file_path)

        # Perform classification
        defect_type = classify_wafer_map_function(wafer_map_tensor, classification_model)
        defect_name = defect_mapping.get(defect_type, "Unknown")

        # Update the global current_defect_type with the latest defect type
        current_defect_type = defect_name  # Storing defect name for clarity in the context

        # Remove the temporary file
        os.remove(temp_file_path)

        # Return the defect type and name
        result = {
            "defect_type": defect_type,
            "defect_name": defect_name
        }
        return result
    except Exception as e:
        # In case of any errors, ensure the temporary file is removed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)