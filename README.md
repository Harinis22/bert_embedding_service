# bert_embedding_service

## Project Description
`bert_embedding_service` is a Streamlit-based web application for log data classification using BERT embeddings. The application allows users to input log data and receive predictions based on a pre-trained BERT model.

## Features
- Log data classification using BERT embeddings
- Interactive web interface with Streamlit
- Data visualization with Altair and Plotly
- Tokenization and preprocessing of log data

## Setup Instructions

### Prerequisites
- Python 3.11
- pip

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/bert_embedding_service.git
    cd bert_embedding_service
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:
    ```sh
    streamlit run src/app.py
    ```

## Usage
1. Open your web browser and go to `http://localhost:8501`.
2. Choose the input type (`Content` or `EventId`).
3. Enter the log data in the text area.
4. Click the "Predict" button to get the classification result.

## Docker Deployment

### Build the Docker Image
1. Build the Docker image:
    ```sh
    docker build -t bert_embedding_service .
    ```

2. Run the Docker container:
    ```sh
    docker run -p 8501:8501 bert_embedding_service
    ```

3. Open your web browser and go to `http://localhost:8501`.

## File Structure
bert_embedding_service/ ├── data/ │ └── Linux_2k.log_structured.csv ├── src/ │ ├── app.py │ ├── model.py │ └── preprocess.py ├── requirements.txt ├── Dockerfile └── README.md

