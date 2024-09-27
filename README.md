# Image Text Extraction and Keyword Highlighting Web App 🖼️🔍

**Author**: Krishna Kaushik

## Project Overview

This project implements an OCR (Optical Character Recognition) solution to extract text from images and allows users to highlight specific keywords within the extracted text. The core functionalities are powered by Huggingface's Qwen2-VL-2B-Instruct model, a vision-language model capable of extracting text in various languages, including Hindi and English. The web interface is developed using **Gradio** to make the process interactive and user-friendly.

⚠️ **Note**: Due to resource limitations on platforms like Hugging Face Spaces (1GB RAM), the text extraction process can take up to **600 seconds** (10 minutes). For faster performance, consider running the application locally on machines with more memory and GPU support.

---

## Features

1. **Image Upload & OCR**:
   - Users can upload images in common formats (JPEG, PNG).
   - The app extracts text from images using the Qwen2-VL-2B-Instruct model, supporting both Hindi and English.

2. **Keyword Search & Highlight**:
   - After extracting text, users can input keywords to search within the extracted text.
   - Keywords are highlighted in the displayed text, making it easier to locate relevant information.

3. **Unicode Support**:
   - The app fully supports Unicode characters, ensuring proper handling of right-to-left languages like Hindi for keyword searches.

---

## Technologies Used

### 1. **Huggingface Transformers**
   - **Model**: Qwen2-VL-2B-Instruct, a powerful model for vision and language tasks, is used to extract text from images. The model handles multilingual text and complex image processing.

### 2. **Gradio**
   - **Interface**: Gradio is used to build the web interface, allowing users to upload images, extract text, and search for keywords. It supports real-time interaction, making the application highly accessible.

### 3. **PyTorch**
   - **Backend**: PyTorch is the core framework powering the Qwen2-VL model. It efficiently handles tensor operations on either CPU or GPU, depending on availability.

### 4. **Pillow (PIL)**
   - **Image Processing**: Pillow is used for processing images uploaded by the user, converting them into a format suitable for model input.

### 5. **Regular Expressions (re)**
   - **Keyword Search**: Python's `re` library is used to search for keywords within the extracted text. It supports case-insensitive and Unicode-aware searches, ensuring accurate keyword matching.

---

## Project Structure

```plaintext
.
├── app.py                # Main application file
├── requirements.txt       # List of dependencies
├── README.md              # This file
└── gradio.yaml            # Gradio deployment config
