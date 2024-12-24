# Running the RAG App Project

This guide outlines the steps to set up and run the RAG (Retrieval-Augmented Generation) application.

## Prerequisites
- Python 3.11 installed.
- Docker and Docker Compose installed.

## Setup Instructions

### 1. Create and Activate a Virtual Environment
Use a virtual environment to manage dependencies for the project.

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Ensure all required Python packages are installed:

```bash
pip install -r requirements.txt
```

### 3. Create env file

```bash
cp .env.example .env
```

add your OPENAI_API_key

### 4. Start the Vector Store
The project uses Qdrant as the vector store. Start it using Docker Compose:

```bash
docker-compose up qdrant-db -d
```

### 5. Load Data
Prepare the data for the application by running the data loading script:

```bash
python3 data_loading.py
```

### 6. Run the Application
You can choose to run either the testing script or the server:

#### Run the Testing Script

```bash
python3 test.py
```

#### Run the Server

```bash
./run_server.sh
```

## Notes
- Ensure the `.env` file is properly configured with the required environment variables.
- The `qdrant-db` service must be running before executing the application scripts.
- If you encounter issues, check logs using:
  ```bash
  docker logs <container_id>
  ```

Feel free to reach out with any questions or concerns.

