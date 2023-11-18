# Assignment_04
----- 
> <br>

> [Codelab Slides] <br>
> [Application Demo/Presentation]

## Index
  - [Objective](#objective)
  - [Project Structure](#project-structure)
  - [How to run the applications](#how-to-run-the-application-locally)
----- 

## Objective
  This application is designed to recommend fashion items based on text descriptions or similar images. It leverages embeddings computed using PINECONE and stores images in Amazon S3. The application is exposed via a FAST API, which supports two main functions: retrieving the closest image based on text input and finding three similar images for a given image. A Streamlit app is provided for user interaction.<br>

  ## Project Structure
```
  fashion-recommendation-app/
│
├── Images/             # Directory to store images
│
├── README.txt          # Project README file
│
├── src/
│   ├── fastapi/
│   │   ├── __init__.py
│   │   ├── main.py    # FAST API implementation
│   │
│   ├── streamlit/
│   │   ├── __init__.py
│   │   ├── main.py    # Streamlit app implementation
│   │   ├── my_utils.py  # Custom utilities
│   │   ├── README.md   # Documentation for Streamlit app
│   │   ├── clip_model.py  # Implementation of CLIP model (assuming it's related to the app)
│   │   ├── requirements.txt  # Requirements for Streamlit app
│
├── search_results.csv  # CSV file containing search results
│
├── requirements.txt    # Project-wide requirements file

## How to run the application
- Clone the repo to get all the source code on your machine

```bash
git clone https://github.com/AlgoDM-Fall2023-Team4/Assignment_04.git
```
- All the code related to the streamlit is in the streamlit/ directory of the project

- First, create a virtual environment, activate and install all requirements from the requirements.txt file present
```bash
python -m venv <virtual_environment_name>
```
```bash
source <virtual_environment_name>/bin/activate
```
```bash
pip install -r <path_to_requirements.txt>
```
- Run the application

```bash
streamlit run streamlit/main.py
```
