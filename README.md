````markdown
# CAD Learning System

A Python-based CAD learning system that processes DXF files and trains an agent to reproduce target shapes.

## Features

- DXF file processing and visualization
- Systematic cutting agent for shape reproduction
- Real-time training visualization using Streamlit
- Progress tracking with similarity metrics

## Installation

1. Create virtual environment:

```bash
python -m venv cad
source cad/bin/activate  # Linux/Mac
# OR
.\cad\Scripts\activate  # Windows
```
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
streamlit run main.py
```

The web interface will open automatically in your default browser.

### Steps:

1. Upload a DXF file using the file uploader
2. Review the initial state and target shape visualization
3. Click "Train Model" to begin the learning process
4. Monitor training progress through the interface

## Project Structure

```
.
├── agents/                 # Agent implementations
├── environment/           # CAD environment simulation
├── utils/                # Utility functions
├── main.py              # Main application
└── requirements.txt     # Package dependencies
```

## Requirements

- Python 3.8+
- streamlit
- numpy
- matplotlib
- scikit-image
- ezdxf

## Development

To export dependencies:

```bash
pip freeze > requirements.txt
```

To install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

```

```
