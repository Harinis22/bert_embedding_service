import streamlit as st
import pandas as pd
import altair as alt
import torch
import numpy as np
import base64
import os
from streamlit_image_select import image_select
from transformers import BertTokenizer, BertForQuestionAnswering
from model import load_model
from preprocess import load_and_preprocess_data, tokenize_data

# Load the trained model and tokenizer
model = load_model()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')  # Use the QuestionAnswering model

# Data Preprocessing
df = load_and_preprocess_data()  # Load and preprocess the data to get the necessary columns
tokenized_data = tokenize_data(df)  # Tokenize the 'combined' column

# Function to convert image to base64
def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error converting image to base64: {str(e)}")
        return None


# Page configuration
st.set_page_config(
    page_title="Log Data Classification Dashboard",
    page_icon= "📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.themes.enable("dark")


# Load the external CSS file
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add a centered layout using the .center class
st.markdown('<div class="center">', unsafe_allow_html=True)

# Displaying the wave animation
st.markdown("""
<div class="waveWrapper">
    <div class="waveWrapperInner bgTop waveAnimation">
        <div class="wave waveTop"></div>
    </div>
    <div class="waveWrapperInner bgMiddle waveAnimation">
        <div class="wave waveMiddle"></div>
    </div>
    <div class="waveWrapperInner bgBottom waveAnimation">
        <div class="wave waveBottom"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Function to convert an image to base64
def img_to_base64(image_path):
    """Convert an image file to a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error converting image: {str(e)}")
        return None

# Image Paths
base_dir = "C:/Users/18136/Desktop/bert/bert_embedding_service/image/"
t_mobile_image = os.path.join(base_dir, "t-mobile.png")
avatar_image = os.path.join(base_dir, "avatar_streamly.png")

# Convert images to Base64
t_mobile_base64 = img_to_base64(t_mobile_image)
avatar_base64 = img_to_base64(avatar_image)

# Sidebar Section with Avatar
with st.sidebar:
    # Title for the sidebar
    st.title('📊 Log Data Classification Dashboard')

    if avatar_base64:
        st.markdown(
             f'<img src="data:image/png;base64,{avatar_base64}" class="glow">', 
            unsafe_allow_html=True
        )
     
    # Dropdown to choose between 'Content' or 'EventId' input
    input_type = st.selectbox("Choose input type", ["Content", "EventId"], key="input_type_selectbox")

    # Data Preview
    st.header("Log Data Preview")
    st.dataframe(df.head())

# Title Section with T-Mobile Logo
if t_mobile_base64:
    st.markdown(
        f"""
        <div class="title-section" style="text-align: center; margin-top: 20px;">
            <h1 style="color: #ff0066;">
                <img src="data:image/png;base64,{t_mobile_base64}"class="glow" alt="T-Mobile Logo" style="display: block; margin: 0 auto; height: 100px;" />
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# Centered Layout for Main Content
col1, col2, col3 = st.columns([1, 3, 1], gap="medium")

with col2:
    st.markdown(
        f"""
        <div class="title-section" style="text-align: center; margin-top: 20px;">
            <h1 style="color: #ff0066;">Log Data Classification </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

 # Randomly select an entry for question-answering
    random_num = np.random.randint(0, len(df))
    question = df["EventId"][random_num]
    text = df["Content"][random_num]
    answer = df["EventTemplate"][random_num]

    st.markdown("""
        <style>
            .card {
                background-color: #ffcee7; /* Light pink background */
                color: ##e20074;  
                padding: 20px;
                border-radius: 8px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 2px solid red;  /* pink border */
            }
            .card-title {
                color: #000000;  /* pink title */
                font-size: 18px;
                font-weight: bold;
            }
            .card-body {
                color: white;
                font-size: 16px;
            }
        </style>
        """, unsafe_allow_html=True)

# Display the question and answer in card format

# Column layout to display question, answer, and tokenization details side by side
    question_col, answer_col = st.columns([1, 1])

    with question_col:
        st.markdown(f'<div class="card"><div class="card-title">Question:</div><div class="card-body">{question}</div></div>', unsafe_allow_html=True)

    with answer_col:
        st.markdown(f'<div class="card"><div class="card-title">Answer (Ground Truth):</div><div class="card-body">{answer}</div></div>', unsafe_allow_html=True)

    # Tokenization and Segment Information
    token_col1, token_col2 = st.columns([1, 1])

with token_col1:
        st.markdown('<div class="card"><div class="card-title">Tokens and IDs:</div><div class="card-body">', unsafe_allow_html=True)
        input_ids = tokenizer.encode(question, text)
        st.write("The input has a total of {} tokens.".format(len(input_ids)))
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Create a DataFrame to display tokens and their corresponding IDs in a table format
        tokens_df = pd.DataFrame(list(zip(tokens, input_ids)), columns=["Token", "ID"])

        # Display the table in Streamlit
        st.dataframe(tokens_df)
        st.markdown('</div></div>', unsafe_allow_html=True)

with token_col2:
        st.markdown('<div class="card"><div class="card-title">Token Segments:</div><div class="card-body">', unsafe_allow_html=True)
        # Find first occurrence of [SEP] token
        sep_idx = input_ids.index(tokenizer.sep_token_id)
        st.write(f"SEP token index: {sep_idx}")

        # Number of tokens in segment A (question)
        num_seg_a = sep_idx + 1
        st.write(f"Number of tokens in segment A (question): {num_seg_a}")

        # Number of tokens in segment B (text)
        num_seg_b = len(input_ids) - num_seg_a
        st.write(f"Number of tokens in segment B (text): {num_seg_b}")

        # Create segment IDs
        segment_ids = [0] * num_seg_a + [1] * num_seg_b
        assert len(segment_ids) == len(input_ids), "Mismatch in token and segment IDs length"
        st.markdown('</div></div>', unsafe_allow_html=True)

# Model Output and Predicted Answer
output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
 # Initialize predicted_answer
predicted_answer = " "
# Tokens with highest start and end scores
EventTemplate_start = torch.argmax(output.start_logits)
EventTemplate_end = torch.argmax(output.end_logits)
# Check if the end token index is greater than or equal to the start token index
if EventTemplate_end >= EventTemplate_start:
        # Join the tokens from start to end
        predicted_answer = tokens[EventTemplate_start]
        for i in range(EventTemplate_start+1, EventTemplate_end+1):
            if tokens[i][0:2] == "##":
                predicted_answer += tokens[i][2:]
            elif tokens[i].startswith("[CLS]"):
                predicted_answer = "Unable to find the answer to your question."
            else:
                predicted_answer += " " + tokens[i]
        print(predicted_answer)           
    
 # Display the predicted answer and content in cards
st.markdown(f'<div class="card"><div class="card-title">Predicted Answer:</div><div class="card-body">{predicted_answer.capitalize()}</div></div>', unsafe_allow_html=True)

    
# Display the Question, Answer, and Predicted Answer in Cards
st.markdown(f'<div class="card"><div class="card-title">Question:</div><div class="card-body">{question.capitalize()}</div></div>', unsafe_allow_html=True)
st.markdown(f'<div class="card"><div class="card-title">Answer:</div><div class="card-body">{answer.capitalize()}</div></div>', unsafe_allow_html=True)
   
