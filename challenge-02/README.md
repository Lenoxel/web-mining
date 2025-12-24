# Challenge 02 - Web Mining

## Setup Instructions

Before executing the `main.py` code for Challenge 02, please ensure that you have created a folder named `data` in the project directory.

### Steps to Follow:

1. Create a folder named `data` in the root of the project directory.
2. Download the `amazon_reviews_2023.parquet` file and place it inside the `data` folder.

**Note:** The `amazon_reviews_2023.parquet` file cannot be uploaded to GitHub due to file size limitations. Please ensure you have this file available in the specified location before running the code.

### Creating a Virtual Environment

Before running the code, it's recommended to create a virtual environment and install the required dependencies:

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:

   - On Linux or macOS:

     ```bash
     source venv/bin/activate
     ```

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

3. Install the required packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

To execute the main code, run:

```bash
python main.py
```

Make sure all dependencies are installed and the data file is in place.