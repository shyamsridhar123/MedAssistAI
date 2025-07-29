"""
MedAssist AI - Clinical Summarization Module
Phase 2.2.2: T5-based Medical Summarization

Implements CPU-optimized T5 medical summarization with proper error handling.
"""

import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalSummarizer:
    """T5-based medical text summarization for clinical notes."""
    
    def __init__(self, model_path: str = None):
        """Initialize the clinical summarizer.
        
        Args:
            model_path: Path to the T5 model directory. If None, uses default path.
        """
        self.model_path = model_path or "models/summarization/model"
        self.tokenizer_path = "models/summarization/tokenizer"
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        
        # CPU optimization settings
        self.device = "cpu"
        self.max_input_length = 512
        self.max_output_length = 150
        self.num_beams = 4
        self.temperature = 0.7
        
    def initialize(self) -> Dict[str, any]:
        """Initialize the T5 model and tokenizer.
        
        Returns:
            Dict with initialization status and details.
        """
        try:
            logger.info("🚀 Initializing T5 Medical Summarization System")
            
            # Check if model files exist
            if not os.path.exists(self.model_path):
                return {
                    "success": False,
                    "error": f"Model path not found: {self.model_path}",
                    "fallback_available": False
                }
            
            logger.info(f"🔽 Loading T5 model from {self.model_path}")
            
            # Load tokenizer
            if os.path.exists(self.tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    local_files_only=True
                )
            else:
                # Fallback to model directory for tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
            
            # Load model with CPU optimization
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch.float32,  # CPU compatibility
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Set number of threads for CPU optimization
            torch.set_num_threads(4)
            
            self.is_initialized = True
            logger.info("✅ T5 Medical Summarizer loaded successfully")
            
            return {
                "success": True,
                "model_type": "T5 Medical Summarization",
                "device": self.device,
                "max_input_length": self.max_input_length,
                "max_output_length": self.max_output_length
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize T5 summarizer: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_available": True
            }
    
    def summarize(self, text: str, max_length: Optional[int] = None) -> Dict[str, any]:
        """Generate clinical summary from medical text.
        
        Args:
            text: Input clinical text to summarize
            max_length: Maximum length of summary (optional)
            
        Returns:
            Dict with summary results and metadata
        """
        if not self.is_initialized:
            init_result = self.initialize()
            if not init_result["success"]:
                return {
                    "success": False,
                    "summary": "",
                    "error": "Model not initialized",
                    "fallback_used": False
                }
        
        try:
            # Prepare input text
            input_text = f"summarize: {text}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=True
            )
            
            # Generate summary
            max_len = max_length or self.max_output_length
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_len,
                    num_beams=self.num_beams,
                    temperature=self.temperature,
                    do_sample=True,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up summary
            summary = summary.strip()
            if not summary:
                summary = self._generate_fallback_summary(text)
                fallback_used = True
            else:
                fallback_used = False
            
            logger.info(f"✅ Summary generated: {len(summary)} characters")
            
            return {
                "success": True,
                "summary": summary,
                "input_length": len(text),
                "output_length": len(summary),
                "fallback_used": fallback_used,
                "model_type": "T5"
            }
            
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            
            # Try fallback summary
            fallback_summary = self._generate_fallback_summary(text)
            
            return {
                "success": False,
                "summary": fallback_summary,
                "error": str(e),
                "fallback_used": True,
                "model_type": "Rule-based fallback"
            }
    
    def _generate_fallback_summary(self, text: str) -> str:
        """Generate a simple rule-based summary as fallback.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Basic extractive summary
        """
        # Simple extractive summarization
        sentences = text.split('. ')
        
        # Keep first sentence and any sentence with key clinical terms
        key_terms = [
            'chief complaint', 'diagnosis', 'treatment', 'condition',
            'patient', 'symptoms', 'medication', 'vital signs'
        ]
        
        important_sentences = []
        
        # Always include first sentence
        if sentences:
            important_sentences.append(sentences[0])
        
        # Add sentences with key clinical terms
        for sentence in sentences[1:]:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in key_terms):
                important_sentences.append(sentence)
                if len(important_sentences) >= 3:  # Limit fallback summary
                    break
        
        summary = '. '.join(important_sentences)
        if not summary.endswith('.'):
            summary += '.'
            
        return summary or "Clinical note summary unavailable."

def test_summarizer():
    """Test function for the clinical summarizer."""
    summarizer = ClinicalSummarizer()
    
    test_text = """
    Patient John Smith was admitted with chest pain and shortness of breath.
    Chief complaint: Acute chest pain radiating to left arm.
    History: 55-year-old male with hypertension and diabetes.
    Physical exam: Blood pressure 160/95, heart rate 88, respiratory rate 20.
    Assessment: Rule out myocardial infarction. EKG shows ST elevation.
    Plan: Administer aspirin, obtain cardiac enzymes, cardiology consult.
    """
    
    print("Testing Clinical Summarizer...")
    init_result = summarizer.initialize()
    print(f"Initialization: {init_result}")
    
    if init_result["success"]:
        result = summarizer.summarize(test_text)
        print(f"Summary Result: {result}")
        print(f"Summary: {result['summary']}")
    else:
        print("❌ Model initialization failed")

if __name__ == "__main__":
    test_summarizer()
