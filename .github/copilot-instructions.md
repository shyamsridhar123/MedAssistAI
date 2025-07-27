# MedAssist AI - Copilot Instructions

## Project Overview
MedAssist AI is an offline clinical assistant for field healthcare deployment, combining Mojo's performance optimization with quantized transformer models for PHI de-identification, clinical summarization, and diagnostic support via a Gradio interface.

## Architecture & Key Concepts

### Hybrid Mojo-Python Pipeline
- **Performance Layer (Mojo)**: Text preprocessing, tokenization, memory management, SIMD operations in `src/`
- **AI Models (Python)**: Hugging Face transformers with 4-bit/8-bit quantization for CPU deployment  
- **UI Layer (Gradio)**: Web interface at `127.0.0.1:7860` with auth `("medassist", "secure123")`
- **Data Flow**: `text → mojo preprocessing → quantized models → gradio output`

### Environment & Dependencies
- Use `magic` (not pip/conda) for environment management: `magic install <package>`
- Mojo project managed via `mojoproject.toml` with MAX platform dependencies
- Key channels: `conda.modular.com/max-nightly`, `conda.modular.com/max`, `conda-forge`
- Python 3.13.5+ required for MAX compatibility

## Development Patterns

### Mojo Integration Conventions
```mojo
# File structure for performance-critical components
from python import Python
from memory import memset_zero
from algorithm import vectorize

# Always use Mojo for:
# - Text preprocessing pipelines
# - Batch processing operations  
# - Memory-intensive operations
# Use Python interop for Hugging Face model integration
```

### Model Quantization Strategy
- **De-identification**: ClinicalBERT with 4-bit quantization (~400MB target)
- **Summarization**: T5-Large/DistilBART with 8-bit + LoRA (~600MB target)  
- **Diagnostics**: MedGemma-4B-it with 4-bit QLoRA (~1.2GB target)
- Performance targets: <30s de-id, <45s summary, <60s diagnosis

### HIPAA Compliance Requirements
- **Offline-only processing** - no external API calls
- **18-identifier removal** with ≥99% recall, ≥95% precision
- **AES-256 encryption** for local storage
- **Configurable retention** with auto-deletion
- **Audit logging** for all PHI processing activities

## Critical Workflows

### Development Environment Setup
```bash
# Initialize project environment
magic install
magic shell  # Activate environment
magic task dev  # Will be defined for development workflow
```

### Model Integration Pattern
```python
# Standard pattern for integrating quantized models
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# CPU-optimized deployment
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="cpu",
    torch_dtype=torch.float16
)
```

### Gradio Interface Standards
- Server binding: `127.0.0.1:7860` (localhost only for security)
- Auth: `("medassist", "secure123")` 
- Three-panel output: de-identified text, clinical summary, diagnostic suggestions
- Export functionality: TXT, PDF, JSON formats
- Retention controls: 0-24 hour configurable auto-deletion

## Data Handling Conventions

### Dataset Integration (see `docs/`)
- **Training data**: i2b2, MIMIC-III, n2c2 datasets for clinical NLP
- **Synthetic data**: Synthea for development/testing without HIPAA concerns
- **Data structure**: `data/train/`, `data/val/`, `data/test/` splits
- **Privacy**: All real clinical data must be pre-de-identified

### File Processing Pipeline
```python
# Standard clinical document processing flow
def process_clinical_note(text, options):
    # 1. Mojo-accelerated preprocessing
    processed = accelerated_text_preprocessing(text)
    
    # 2. Sequential model pipeline
    if options.deid_enabled:
        processed = deid_model(processed)
    if options.summary_enabled:
        summary = summary_model(processed)  
    if options.diag_enabled:
        diagnosis = diagnostic_model(processed)
        
    return structured_output
```

## Performance Optimization

### CPU Optimization Focus
- Target: Consumer hardware (8GB RAM minimum, 16GB recommended)
- Leverage Mojo's SIMD vectorization for text operations
- Quantized models for 75% memory reduction vs full precision
- Multi-core batch processing without GIL constraints

### Memory Management
- Peak usage <4GB during processing 
- Automatic cleanup of sensitive data after configurable timeout
- Use Mojo's ownership system for efficient allocation/deallocation

## Testing & Validation

### Clinical Accuracy Benchmarks
- **PHI Detection**: i2b2 2014 dataset, target F1-score ≥0.9732
- **Summarization**: Clinical professional validation for accuracy
- **Diagnostics**: Evidence-based recommendations with confidence scoring

### Security Testing
- Offline operation validation (no network calls)
- PHI removal verification (18 HIPAA identifiers)
- Encryption verification for local storage
- Audit trail completeness

## Common Pitfalls & Guidelines

- **Never** add internet connectivity or external API calls
- **Always** use quantized models for CPU deployment  
- **Encrypt** any persistent storage of clinical data
- **Log** all PHI processing for audit compliance
- Use `magic` not `pip` for dependency management
- Prefer Mojo for performance-critical text processing
- Maintain offline-first architecture throughout

## Key Files & Entry Points
- `main.mojo`: Application entry point
- `src/python_integration.mojo`: Mojo-Python model integration  
- `mojoproject.toml`: Environment and dependency configuration
- `docs/Product Requirements Document (PRD) & Technical Sp.md`: Complete technical specification
