import gradio as gr
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch
import re

# Load the pre-trained model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Function to extract text from the image
def extract_text(image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "can u extract the text in hindi"}
            ]
        }
    ]
    
    # Process input image and text prompt
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate output text from the model
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    
    # Decode the generated text
    extracted_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]  # Extracted text
    
    return extracted_text

# Function to highlight keywords in the text, even for right-to-left scripts like Hindi
def highlight_keywords(extracted_text, keywords):
    highlighted_text = extracted_text
    if keywords:
        for keyword in keywords.split(","):
            keyword = keyword.strip()
            if keyword:
                # Ensure correct Unicode support for keywords (use re.UNICODE for non-ASCII)
                highlighted_text = re.sub(
                    re.escape(keyword),  # Use re.escape to handle special characters in keywords
                    r'<mark>\g<0></mark>',  # Highlight the found keyword
                    highlighted_text,
                    flags=re.IGNORECASE | re.UNICODE  # Ignore case, and handle Unicode characters
                )
    
    return highlighted_text

# First step: Extract text from the uploaded image
def extract_text_step(image):
    extracted_text = extract_text(image)
    return extracted_text, extracted_text  # Return extracted text and store it in state

# Second step: Search and highlight keywords in the extracted text
def highlight_keywords_step(extracted_text, keywords):
    highlighted_text = highlight_keywords(extracted_text, keywords)
    return highlighted_text

# Gradio UI
with gr.Blocks() as demo:
    # Step 1: Image Upload and Text Extraction
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        extract_button = gr.Button("Extract Text")
    extracted_text_output = gr.Textbox(label="Extracted Text")
    
    # Step 2: Keyword Input and Highlighting
    with gr.Row():
        keyword_input = gr.Textbox(label="Enter keywords (comma-separated)", placeholder="Enter keywords after text extraction")
        search_button = gr.Button("Highlight Keywords")
    highlighted_text_output = gr.HTML(label="Highlighted Text with Matches")
    
    # Define interactions
    extract_button.click(
        fn=extract_text_step,  # Call text extraction function
        inputs=image_input,
        outputs=[extracted_text_output, extracted_text_output],  # Display text and store in state
    )
    
    search_button.click(
        fn=highlight_keywords_step,  # Call keyword highlighting function
        inputs=[extracted_text_output, keyword_input],  # Use extracted text and keywords
        outputs=highlighted_text_output,  # Display highlighted text
    )

# Launch the app
demo.launch()
