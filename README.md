# Chat with Your CSV

This Streamlit application allows users to upload a CSV file and interact with its data using natural language queries. The app uses OpenAI's GPT model to interpret user questions and generate appropriate responses, including visualizations and data analysis.

## Features

- CSV file upload and parsing
- Natural language interaction with CSV data
- Various data visualization options (tables, charts, graphs)
- Conversation memory for context-aware responses

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/chat-with-csv.git
   cd chat-with-your-csv
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Enter your OpenAI API key when prompted.

4. Upload a CSV file using the file uploader in the sidebar.

5. Start asking questions about your data in natural language.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.