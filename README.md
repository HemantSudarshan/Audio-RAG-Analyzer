


### README.md

![Screenshot 2025-03-08 221008](https://github.com/user-attachments/assets/8a338a6d-e65b-43bf-a237-99bd58600d48)

![Json](https://github.com/user-attachments/assets/dc7980d2-c9f6-4a05-83bc-92a527f0edc8)



```markdown
# Audio RAG Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red) ![License](https://img.shields.io/badge/License-MIT-green)

A sophisticated tool for transcribing audio, performing sentiment analysis, and querying content using Retrieval-Augmented Generation (RAG). Built with Python and Streamlit, this project provides an interactive, black-themed UI with cyan-blue accents for processing audio files, generating summaries, and exporting results.

## Features
- **Audio Transcription**: Converts audio files (MP3, WAV, M4A) to text using AssemblyAI.
- **Sentiment Analysis**: Analyzes transcript emotions with TextBlob.
- **RAG Integration**: Queries audio content using DeepSeek-R1-Distill-Llama-70B.
- **Interactive UI**: Streamlit interface with icon-based tabs (⚙️ Process, 📜 Transcript, 💬 Chat).
- **Export Options**: Save transcripts as PDF or JSON.
- **Statistics**: Displays segment count, average word count, sentiment scores, and unique speakers.
- **Chat History**: Save and export conversation history with a sleek chat interface.

## Screenshots
![UI Screenshot](path/to/screenshot.png) *Black-themed UI with cyan-blue buttons and animated chat.*

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/audio-rag-analyzer.git
   cd audio-rag-analyzer
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment**:
   - Create a `.env` file in the root directory.
   - Add your AssemblyAI API key:
     ```
     ASSEMBLYAI_API_KEY=your_api_key_here
     ```
4. **Run the App**:
   ```bash
   streamlit run HEMP4.py
   ```

## Requirements
- Python 3.8+
- Libraries: `streamlit`, `textblob`, `fpdf`, `python-dotenv`, `pandas`, `rag_code` (custom module)
- AssemblyAI API key

## Usage
1. Upload an audio file via the sidebar.
2. Select language and export formats (PDF/JSON).
3. Process audio to view transcripts, stats, and optional summaries.
4. Use the chat tab (💬) to query audio content.
5. Export transcripts or chat history as needed.

## Project Structure
```
audio-rag-analyzer/
├── HEMP4.py         # Main application code
├── .env            # Environment variables (API keys)
├── requirements.txt # Dependencies
└── README.md       # This file
```

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [AssemblyAI](https://www.assemblyai.com/) for audio transcription API.
- [Streamlit](https://streamlit.io/) for the interactive UI framework.
- [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis.

---
Created by [Hemant Sudarshan] | [LinkedIn](https://linkedin.com/in/yourprofile) | March 2025
```

---
