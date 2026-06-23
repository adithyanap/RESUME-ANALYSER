# RESUME-ANALYSER


Job Title Predictor
Predicts job titles from PDF resumes using an LSTM model.

Project Description
This project is an automated machine learning pipeline that maps professional skills to specific job titles. It functions by reading a dataset of job roles and requirements, training a Deep Learning sequence model, and extracting text from a user-provided PDF resume. By treating a candidate's background as a sequence of technical terms, the system maps the parsed resume text against trained patterns to recommend the most relevant engineering or corporate position.

Key Features
Automated Data Preprocessing: Automatically standardizes all data sources by converting text to lowercase, eliminating syntax variation, and handling vocabulary mapping.

Custom PDF Parsing Engine: Extracts structured text blocks from multi-page PDFs, dynamically converting noisy punctuation marks (like commas, slashes, and symbols) into clean term delimiters.

Dual Vocabulary Customization: Configures separate vocabulary limits for input sequences (up to 10,000 unique skills) and target classifications (up to 3,000 unique job positions) to keep processing efficient.

Deep Learning Sequence Matching: Utilizes an Embedding and LSTM (Long Short-Term Memory) architecture to capture the context and combination of skills rather than just looking at isolated keywords.

Pipeline Serialization: Automatically saves the trained Keras model (.h5) and the exact tokenizer configurations (.pkl) so the script can immediately load existing assets for live predictions without retraining every time.
