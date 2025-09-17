# CUBO Desktop GUI

A professional, personalizable desktop interface for the CUBO RAG system.

## Features

### 🎨 **Professional Interface**

- Native desktop application (no browser required)
- Clean, enterprise-grade design
- Customizable themes (Light, Dark, Corporate, High Contrast)
- Responsive layout with professional typography

### 📁 **Document Management**

- Drag-and-drop document upload
- Support for PDF, DOCX, and TXT files
- Real-time processing with progress indicators
- Document list with metadata display

### 🔍 **Query Interface**

- Natural language query input
- Real-time AI response generation
- Source document citations
- Response history and export options

### ⚙️ **Settings & Configuration**

- LLM model selection (Ollama integration)
- Document processing parameters
- Performance tuning options
- Theme customization

### 📊 **Analytics Dashboard**

- System status monitoring
- Query performance metrics
- Usage statistics
- Vector database health

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Ensure Ollama is running with your preferred models:

```bash
ollama serve
ollama pull llama3.2  # or your preferred model
```

## Usage

### Launch the GUI

```bash
python launch_gui.py
```

### Alternative Launch Methods

```bash
# Direct module execution
python -m gui.main_window

# From Python
from gui.main_window import main
main()
```

## Interface Overview

### Main Window

- **Menu Bar**: File, View, Help menus
- **Toolbar**: Quick access to common functions
- **Status Bar**: Real-time system status and progress
- **Tabbed Interface**: Organized into 4 main sections

### Document Tab

- Upload documents via file browser or drag-and-drop
- View processing progress
- Manage loaded documents
- Delete documents from the knowledge base

### Query Tab

- Enter natural language questions
- View AI-generated responses
- See source document citations
- Access query history

### Settings Tab

- Configure LLM models and parameters
- Adjust document processing settings
- Customize performance options
- Save/load configuration profiles

### Analytics Tab

- Monitor system health
- View query performance metrics
- Track usage statistics
- Access system logs

## Personalization

### Themes

Choose from built-in themes via View → Theme menu:

- **Light**: Clean, professional appearance
- **Dark**: Easy on the eyes for extended use
- **Corporate**: Blue-based enterprise styling
- **High Contrast**: Accessibility-focused design

### Configuration

Settings are automatically saved to `config.json` and persist between sessions.

## Backend Integration

The GUI seamlessly integrates with CUBO's backend:

- **Document Processing**: Uses `DocumentLoader` for file parsing
- **Vector Search**: Leverages `DocumentRetriever` for similarity search
- **Response Generation**: Calls `ResponseGenerator` for AI responses
- **Configuration**: Reads from and writes to `config.json`
- **Logging**: All operations logged via the backend logger

## Requirements

- Python 3.11+
- PySide6 6.5.0+
- Ollama running locally
- All CUBO backend dependencies

## Troubleshooting

### GUI Won't Start

- Ensure PySide6 is installed: `pip install PySide6`
- Check Python path includes the `src` directory
- Verify backend modules are accessible

### Backend Connection Issues

- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Verify vector database is accessible

### Performance Issues

- Adjust chunk size in Settings (smaller = faster but less context)
- Enable/disable GPU usage based on hardware
- Monitor memory usage in Analytics tab

## Development

### Project Structure

```
gui/
├── main_window.py    # Main application window
├── components.py     # UI component widgets
├── themes.py         # Theme management system
└── dialogs.py        # Modal dialogs
```

### Adding New Features

1. Extend components in `components.py`
2. Add theme support in `themes.py`
3. Connect signals in `main_window.py`
4. Update backend integration as needed

## Enterprise Features

- **Offline Operation**: No internet required
- **Data Privacy**: All processing local
- **Professional UI**: Enterprise-grade interface
- **Configurable**: Adaptable to corporate branding
- **Scalable**: Handles large document collections
- **Auditable**: Comprehensive logging and monitoring

---

**Ready for Enterprise Deployment** 🚀

This GUI transforms CUBO from a command-line tool into a professional desktop application suitable for enterprise users paying €5/month for premium RAG capabilities.
