FROM python:3.11-slim

# Create a non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download NLTK data required by the pipeline
RUN python -m nltk.downloader stopwords punkt punkt_tab

# Copy the rest of the application
COPY --chown=user . /app

# Run the Flask app
CMD ["python", "app.py"]
