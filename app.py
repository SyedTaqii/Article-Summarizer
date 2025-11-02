import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- 1. Load Model and Tokenizer (Cached) ---
# This decorator is the most important part.
# It tells Streamlit to load the model into memory only ONCE,
# making the app much faster after the first run.
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned model and tokenizer from the local directory.
    """
    model_path = 'syedtaqi/article-summarizer'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    st.info(f"Loading model onto {device.upper()}... (This will only happen once)")
    
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the 't5_final_summarizer_model' folder is in the same directory as app.py")
        return None, None, None
        
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    st.success("Model loaded successfully!")
    return model, tokenizer, device

# Load the model once at the start
model, tokenizer, device = load_model()

# --- 2. The Inference Function ---
def generate_summary(article_text):
    """
    Generates a summary for a given article text.
    """
    if not model or not tokenizer:
        return "Error: Model or Tokenizer not loaded."

    # T5 requires the "summarize: " prefix
    PREFIX = "summarize: "
    input_text = PREFIX + article_text
    
    # Tokenize the article
    encoding = tokenizer(
        input_text, 
        max_length=512,  # Max input length (from training)
        truncation=True, 
        return_tensors="pt"
    )
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    
    # Generate the summary
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,  # Max length for the generated summary
            num_beams=4,          # Use beam search for higher quality
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated summary
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

# --- 3. The Streamlit UI ---
st.title("📰 Article Text Summarization")
st.markdown(
    "Paste in a long news article below to get a concise summary."
)

st.subheader("Enter Your Article")

# Use a text_area for multi-line input
article_input = st.text_area(
    "Paste the full text of the article here:", 
    height=300,
    placeholder="Your Article..."
)

# Generate button
if st.button("Generate Summary"):
    if not model:
        st.error("Model is not loaded. Please check your file path and restart.")
    elif not article_input.strip():
        st.warning("Please paste an article into the text box.")
    else:
        # Show a spinner while the model works
        with st.spinner("Generating your summary..."):
            summary_output = generate_summary(article_input)
            
            st.subheader("Generated Summary")
            st.success(summary_output) # Show the summary in a green box

# st.sidebar.header("About This Project")
# st.sidebar.info(
#     "**Task 2: Encoder-Decoder Model**\n\n"
#     "This app fulfills the second part of the assignment, demonstrating a "
#     "fine-tuned T5 model for abstractive summarization.\n\n"
#     f"**Model:** `t5-small`\n"
#     f"**Dataset:** CNN/DailyMail (50k sample)\n"
#     f"**ROUGE-1:** 42.6%"
# )
