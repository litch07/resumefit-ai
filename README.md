<h1 align="center">ResumeFit AI</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Flask-3.1.3-black.svg" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg" alt="Hugging Face">
</p>

ResumeFit AI analyzes resumes against job descriptions to provide a compatibility score and skill gap analysis.

**Live Demo**: [https://huggingface.co/spaces/litch07/resumefit-ai](https://huggingface.co/spaces/litch07/resumefit-ai)

## Features

- Match score from 0 to 100 showing how well the resume fits the job description
- Missing keyword detection highlighting terms from the job description not found in the resume
- Improvement suggestions as a prioritized action list
- Job role prediction showing the top 3 most likely categories based on resume content
- Support for PDF, DOCX, and TXT resume formats
- No sign-up, no data storage, no external APIs

## How It Works

1. **Upload** — the user uploads a resume (PDF, DOCX, or TXT) and pastes a job description into the text area.
2. **Parse** — the system extracts raw text from the file using pdfplumber, PyPDF2, or python-docx.
3. **Analyse** — SBERT generates semantic embeddings for both texts and cosine similarity produces a match score.
4. **Keywords** — TF-IDF and RAKE extract important terms from the job description and identify which are missing from the resume.
5. **Results** — the UI shows the match score, missing keywords, improvement suggestions, and predicted job roles — all without a page reload.

## Tech Stack

| Category | Tools |
| :--- | :--- |
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Flask 3.1.3, Werkzeug 3.1.8 |
| **Machine Learning** | scikit-learn 1.8.0, scipy 1.17.1, joblib 1.5.3 |
| **NLP** | nltk 3.9.4, rake-nltk 1.0.6, sentence-transformers 5.5.1 |
| **Data Processing** | numpy 2.4.6, pandas 3.0.3, torch 2.12.0 |
| **Document Parsing** | pdfplumber 0.11.9, PyPDF2 3.0.1, python-docx 1.2.0, Pillow 12.2.0 |

## Quick Start

### Prerequisites
- Python 3.11+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/litch07/resumefit-ai.git
   cd resumefit-ai
   ```

2. **Create and activate a virtual environment**
   
   *Windows (PowerShell):*
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   
   *Mac/Linux (bash):*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -m nltk.downloader stopwords punkt punkt_tab
   ```

5. **Download Model Files**
   Download the model files from GitHub Releases as described in the next section.

6. **Run the application**
   ```bash
   python app.py
   ```
   *Note: The first run automatically downloads the SBERT model (~90MB).*

The app runs at `http://localhost:5000`.

## Download Model Files

The trained machine learning models are distributed via GitHub Releases. They are not committed to the repository due to file size constraints.

Link: [https://github.com/litch07/resumefit-ai/releases/latest](https://github.com/litch07/resumefit-ai/releases/latest)

Download these three files and place them in the `models/` directory:
- `job_classifier.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

## Training Your Own Model (Optional)

Users who want to retrain the model can use the included training script. The default model was trained on the Kaggle resume dataset. Training data is not included in this repository due to its size.

Before training, the dataset requires cleaning to remove corrupted rows. See the data preparation steps in the project documentation at `/docs` once the app is running.

To train the model, run:
```bash
python src/trainer.py
```

## Project Structure

```text
resumefit-ai/
├── app.py                    # Flask entry point and route definitions
├── requirements.txt          # Pinned package dependencies
├── README.md
├── API.md                    # HTTP API reference
├── Dockerfile                # Container configuration for deployment
├── .dockerignore             # Files excluded from Docker builds
├── .gitignore
├── models/                   # Downloaded .pkl model files (not in Git)
│   ├── job_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── src/
│   ├── parser.py             # Extracts text from PDF, DOCX, TXT
│   ├── preprocessor.py       # Cleans and normalises text
│   ├── keyword_extractor.py  # TF-IDF and RAKE keyword extraction
│   ├── embedder.py           # SBERT embeddings and similarity
│   ├── scorer.py             # Main analysis orchestrator
│   ├── predictor.py          # Job role prediction
│   └── trainer.py            # Standalone model training script
├── static/
│   ├── style.css
│   └── script.js
└── templates/
    ├── index.html
    ├── docs.html
    ├── privacy.html
    ├── security.html
    └── terms.html
```

## Model Performance

The job classifier is a LinearSVC model trained on 2,457 resumes across 24 job categories using an 80/20 stratified train/test split. It uses dual TF-IDF features (word-level and character-level combined with `scipy.sparse.hstack`). Hyperparameter search confirmed `C=1.0` is optimal after testing `C=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`.

The predictor applies softmax normalization to the top N scores. This ensures confidence percentages are meaningful rather than returning low raw margins like 5-7%. Role names are returned in Title Case.

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | 71.75% |
| **CV Macro F1 Mean** | 0.659 (5-fold) |
| **Best Category (DESIGNER)** | F1 0.89 |

*Note: The BPO category achieved an F1 score of 0.00. This limitation is due to insufficient samples in the dataset, with only 4 test samples available for this category.*

## Deployment

The application uses Docker for deployment to Hugging Face Spaces. The included `Dockerfile` installs all system requirements and Python dependencies, then starts the Flask application. The `.dockerignore` file prevents local models and virtual environments from bloating the image.

The `PORT` environment variable is read at startup, so the same image runs on local (port 5000) and Hugging Face Spaces (port 7860) without any code changes.

## Troubleshooting

- **Missing Model Files**: Ensure `job_classifier.pkl`, `tfidf_vectorizer.pkl`, and `label_encoder.pkl` are located in the `models/` directory.
- **Port 5000 Already in Use**: Stop other services using the port, or change the port in `app.py`.
- **NLTK Data Errors**: Run the NLTK download command listed in the Quick Start section.
- **High Memory Usage on First Run**: The SBERT model downloads on the first execution. Ensure you have a stable internet connection and sufficient free RAM.

## Future Work

- Interactive Editing and Recalculation
- PDF Export of gap analysis report

## License

This project is licensed under the [MIT License](LICENSE).
