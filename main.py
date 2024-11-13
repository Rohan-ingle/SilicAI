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

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import asyncio
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
import json
import os
import warnings
from langchain.schema import Document

import fitz

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variable for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins (or specify which origins are allowed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods like GET, POST, OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Load PDF using PyMuPDF (fitz)
def load_pdf_with_fitz(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Load the document (you can upload or specify a path here)
local_path = "context.pdf"
pdf_text = load_pdf_with_fitz(local_path) if local_path else ""

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Wrap the PDF text into a Document
documents = [Document(page_content=pdf_text)]  # Create Document object with 'page_content'

# Split the document into chunks
chunks = text_splitter.split_documents(documents)

# Create vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="local-rag"
)

local_model = "llama3.2:1b"  # Or whichever model you prefer
llm = ChatOllama(model=local_model)

# Prompt template for handling various types of questions
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are SilicAI, a QnA Chatbot that helps users with semiconductor defect analysis and general knowledge. 
When the user asks a greeting or personal question, respond appropriately.
When the user asks about the nature of defects in semiconductor manufacturing or technical aspects, respond based on the context from the PDF document provided.

- If the user greets, respond with a polite greeting.
- If the user asks 'who are you' or 'what do you do', provide a brief introduction about yourself.
- If the user asks any technical or document-related question, provide an answer based on the context of the PDF document.

Question: {question}
Answer: """,
)

# Set up retriever to generate variations of the question
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# Define the LLM's response template more concisely
template = """Answer the question based ONLY on the following context and do not repeat the setup or unnecessary details except unrelated to question and answering like greetings:
{context}
Question: {question}
Answer concisely based on context only.
"""

prompt = ChatPromptTemplate.from_template(template)

# Final chain for processing the question and retrieving the answer
chain = (
    {"context": retriever, "question": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)

# Pydantic model to receive input as JSON
class ChatRequest(BaseModel):
    question: str
    defect: str = ""  # Keep defect as optional field

# Asynchronous generator to process the token stream
async def process_tokens(stream):
    queue = Queue()

    def run_stream():
        try:
            for token in stream:
                # Attempt to access the correct attribute
                # Adjust attribute names as necessary
                token_text = getattr(token, 'content', None)
                if token_text is None:
                    token_text = getattr(token, 'text', None)
                if token_text is None:
                    token_text = str(token)
                # Convert to JSON string, append newline
                json_str = json.dumps({"token": token_text}) + "\n"
                queue.put_nowait(json_str)
        except Exception as e:
            json_str = json.dumps({"error": str(e)}) + "\n"
            queue.put_nowait(json_str)
        finally:
            queue.put_nowait(None)  # Signal completion

    # Run the synchronous stream in a background thread
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, run_stream)

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item.encode('utf-8')  # Yield bytes

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """
    Endpoint for chatting with the PDF.
    Accepts a question and defect information, and returns an answer from the PDF.
    """
    formatted_question = request.question.strip()

    try:
        # Pass the question to llm.stream as a string or a BaseMessage object
        # Ensure that the input type matches what llm.stream() expects
        stream = llm.stream(formatted_question)  # Pass the question as a string
        return StreamingResponse(process_tokens(stream), media_type="application/json")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)  # Handle errors and return as response

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
    0: "[['none']]",
    1: "[['Loc']]",
    2: "[['Edge-Loc']]",
    3: "[['Center']]",
    4: "[['Edge-Ring']]",
    5: "[['Scratch']]",
    6: "[['Random']]",
    7: "[['Near-full']]",
    8: "[['Donut']]"
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
SEGMENTATION_MODEL_PATH = "segmentation_model.pth"
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

# Define the /ml endpoint for classification
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