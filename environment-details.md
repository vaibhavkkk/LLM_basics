# Environment.yml Configuration Guide

This document explains the `environment.yml` file used to set up a Python environment for Large Language Models (LLMs) and machine learning development.

## Overview

The `environment.yml` file is a Conda environment configuration that creates a comprehensive Python environment named `vk-llms` for working with various LLM providers, machine learning libraries, and data science tools.

## Environment Configuration

### Environment Name
```yaml
name: vk-llms
```
Creates a named Conda environment that can be activated/deactivated independently from other Python environments.

### Channels
```yaml
channels:
  - conda-forge
  - defaults
```
- **conda-forge**: Community-driven package repository with the latest versions and better maintenance
- **defaults**: Anaconda's official default package channel

## Dependencies Structure

The file organizes dependencies into two categories:

### 1. Conda Dependencies
These packages are installed directly via Conda package manager:

```yaml
dependencies:
  - python=3.11
  - pip
  - python-dotenv
  - requests
  - numpy
  - pandas
  - scipy
  - pytorch
  - jupyterlab
  - ipywidgets
  - matplotlib
  - scikit-learn
  - chromadb
  - jupyter-dash
  - pyarrow
```

**Core Components:**
- **Python 3.11**: The Python interpreter version
- **Scientific Computing**: numpy, pandas, scipy for data manipulation and numerical computing
- **Machine Learning**: scikit-learn for traditional ML algorithms
- **Deep Learning**: pytorch for neural networks and deep learning
- **Development Environment**: jupyterlab, ipywidgets for interactive notebooks
- **Data Storage**: pyarrow for efficient data serialization, chromadb for vector databases
- **Visualization**: matplotlib for plotting and data visualization

### 2. Pip Dependencies
These packages are installed via pip within the Conda environment:

```yaml
pip:
  - beautifulsoup4
  - plotly
  - transformers
  - sentence-transformers
  - datasets==3.6.0
  - openai
  - anthropic
  - google-generativeai
  - gradio
  - gensim
  - modal
  - ollama
  - psutil
  - setuptools
  - speedtest-cli
  - langchain
  - langchain-core
  - langchain-text-splitters
  - langchain-openai
  - langchain-chroma
  - langchain-community
  - feedparser
  - twilio
  - pydub
  - protobuf==3.20.2
```

**Key Package Categories:**

#### LLM and AI Libraries
- **transformers**: Hugging Face transformers library for pre-trained models
- **sentence-transformers**: Specialized library for sentence embeddings
- **datasets**: Hugging Face datasets library (pinned to version 3.6.0)
- **gensim**: Topic modeling and document similarity analysis

#### LLM Provider APIs
- **openai**: OpenAI API client (GPT models)
- **anthropic**: Anthropic API client (Claude models)
- **google-generativeai**: Google's Generative AI API client (Gemini models)

#### Local Model Inference
- **ollama**: Run large language models locally

#### LangChain Ecosystem
- **langchain**: Core LangChain library for LLM application development
- **langchain-core**: Core abstractions and interfaces
- **langchain-text-splitters**: Text chunking utilities
- **langchain-openai**: OpenAI integration
- **langchain-chroma**: ChromaDB integration
- **langchain-community**: Community-contributed integrations

#### Application Development
- **gradio**: Create web interfaces for ML models
- **modal**: Cloud deployment and serverless computing
- **twilio**: SMS and communication services
- **feedparser**: RSS/Atom feed parsing

#### Utilities
- **beautifulsoup4**: HTML/XML parsing
- **plotly**: Interactive plotting library
- **psutil**: System and process utilities
- **speedtest-cli**: Internet speed testing
- **pydub**: Audio manipulation
- **protobuf**: Protocol buffers (pinned to version 3.20.2 for compatibility)

## Usage Instructions

### Creating the Environment
```bash
# Create the environment from the file
conda env create -f environment.yml
```

### Activating the Environment
```bash
# Activate the environment
conda activate vk-llms
```

### Deactivating the Environment
```bash
# Deactivate when done
conda deactivate
```

### Updating the Environment
```bash
# Update environment with changes to environment.yml
conda env update -f environment.yml --prune
```

### Removing the Environment
```bash
# Remove the entire environment
conda env remove -n vk-llms
```

## Use Cases

This environment is designed for:

1. **Multi-Provider LLM Development**: Work with OpenAI, Anthropic, and Google models
2. **Local Model Inference**: Run models locally using transformers and ollama
3. **RAG Applications**: Build Retrieval-Augmented Generation systems with LangChain and ChromaDB
4. **Data Science Workflows**: Analyze and visualize data with pandas, numpy, and matplotlib
5. **Interactive Development**: Use Jupyter notebooks for experimentation
6. **Application Deployment**: Create web interfaces with Gradio and deploy with Modal
7. **Audio/Media Processing**: Handle audio files with pydub
8. **Web Scraping**: Parse web content with beautifulsoup4
9. **Communication Integration**: Send SMS/notifications with Twilio

## Version Pinning

Note that some packages are pinned to specific versions:
- `datasets==3.6.0`: Ensures compatibility with specific dataset formats
- `protobuf==3.20.2`: Prevents conflicts with TensorFlow and other libraries

This comprehensive setup provides everything needed for modern LLM development and deployment workflows.
