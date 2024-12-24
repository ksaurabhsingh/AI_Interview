# AI Bot Interview Application

## Overview
The AI Voice Interview Application is an advanced AI-based system designed to conduct voice interviews, evaluate candidates, and generate scores along with remarks and attributes. It uses state-of-the-art technologies, including text-to-speech, speech-to-text, and natural language processing (NLP), to streamline and automate the interview process.

## Key Features
1. **Resume Parsing and Question Generation**  
   - Upload resumes in PDF or Word format.  
   - Extract relevant information using RAG (Retrieval-Augmented Generation) and map-reduce techniques.  
   - Generate meaningful and tailored interview questions based on the resume content.  

2. **AI-Powered Voice Interviews**  
   - Text-to-voice conversion to ask questions via AI.  
   - Capture voice responses and convert them to text for analysis.  

3. **Summarization and Evaluation**  
   - Summarize answers using NLP models.  
   - Evaluate responses using Llama 3.2 with pre-defined prompts.  
   - Provide scores, remarks, and highlight positive attributes.  

4. **User Interface**  
   - Flask and Streamlit-based web application for an interactive user experience.  
   - Upload resumes, monitor interviews, and download results in Excel format.  

## Technologies Used
- **Programming Languages**: Python  
- **Frameworks**: Flask, Streamlit  
- **NLP Models**: Llama 3.2, LangChain  
- **Speech Processing**: Coqui TTS, PyAudio, Silero Models  
- **Storage and Output**: Excel for storing results  

## Architecture
1. **Frontend**: Built using Streamlit and Flask for seamless user interaction.  
2. **Backend**: Python-based processing pipeline for voice capture, transcription, and analysis.  
3. **AI Models**: NLP and summarization models integrated via LangChain and Ollama.  
4. **Evaluation Module**: AI-based scoring and feedback generation.  

## Setup Instructions
### Prerequisites
1. Python 3.9 or higher  
2. Pip or Conda package manager  
3. Virtual environment (recommended)  

### Installation Steps
1. Clone the repository:
```
git clone <repository-url>
```
2. Navigate to the project directory:
```
cd ai_interview
```
3. Create a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate    # For Windows
```
4. Install dependencies:
```
pip install -r requirements.txt
```
5. Start the application:
```
streamlit run app.py
```

## Usage
1. Open the web application in a browser:
```
http://localhost:8501
```
2. Upload the resume file (PDF/Word).  
3. Generate interview questions.  
4. Start the voice interview.  
5. Download the interview summary report as an Excel file.  

## Output Format
- **Excel Report**: Includes responses, scores, remarks, and highlighted strengths of the candidate.  
- **Logs**: Debugging logs stored locally for troubleshooting.  

## Future Enhancements
- Multi-language support for interviews.  
- Integration with video conferencing tools.  
- Advanced analytics for performance tracking.  

## Contributing
Contributions are welcome! Please create a pull request or raise issues in the repository for enhancements or bug fixes.


## Author
**Saurabh Singh**  
AI Developer / Data Scientist

