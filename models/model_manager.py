"""
MedAssist AI - Unified Model Manager
Handles loading and coordination of all clinical AI models

Architecture:
- PHI Detection: ClinicalBERT + rule-based hybrid
- Summarization: T5 medical summarization model  
- Diagnostics: BioGPT (future implementation)
"""

import logging
import os
import sys
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Unified manager for all clinical AI models."""
    
    def __init__(self, models_root: str = "models"):
        """Initialize the model manager.
        
        Args:
            models_root: Root directory containing all model subdirectories
        """
        self.models_root = models_root
        self.loaded_models = {}
        self.model_configs = {
            "phi_detection": {
                "module_path": "phi_detection.hybrid_detector",
                "class_name": "HybridPHIDetector",
                "enabled": True
            },
            "summarization": {
                "module_path": "summarization.clinical_summarizer", 
                "class_name": "ClinicalSummarizer",
                "enabled": True
            },
            "diagnostics": {
                "module_path": "diagnostics.clinical_diagnostics",
                "class_name": "ClinicalDiagnostics", 
                "enabled": True  # Now implemented with MedGemma 4B GGUF
            }
        }
        
        # Add models directory to Python path
        if models_root not in sys.path:
            sys.path.insert(0, models_root)
            
        logger.info(f"🚀 ModelManager initialized with root: {models_root}")
    
    def load_model(self, model_type: str) -> Dict[str, Any]:
        """Load a specific model type.
        
        Args:
            model_type: Type of model ('phi_detection', 'summarization', 'diagnostics')
            
        Returns:
            Dict with loading results and model instance
        """
        if model_type not in self.model_configs:
            return {
                "success": False,
                "error": f"Unknown model type: {model_type}",
                "model": None
            }
            
        config = self.model_configs[model_type]
        
        if not config["enabled"]:
            return {
                "success": False,
                "error": f"Model {model_type} is disabled",
                "model": None
            }
            
        if model_type in self.loaded_models:
            logger.info(f"✅ Model {model_type} already loaded")
            return {
                "success": True,
                "model": self.loaded_models[model_type],
                "cached": True
            }
            
        try:
            logger.info(f"🔽 Loading {model_type} model...")
            
            # Dynamic import
            module_name = config["module_path"]
            class_name = config["class_name"]
            
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # Instantiate model
            model_instance = model_class()
            
            # Initialize if method exists
            if hasattr(model_instance, 'initialize'):
                init_result = model_instance.initialize()
                if not init_result.get("success", True):
                    return {
                        "success": False,
                        "error": f"Model initialization failed: {init_result.get('error', 'Unknown error')}",
                        "model": None
                    }
            
            self.loaded_models[model_type] = model_instance
            logger.info(f"✅ {model_type} model loaded successfully")
            
            return {
                "success": True,
                "model": model_instance,
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_type}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": None
            }
    
    def process_phi_detection(self, text: str) -> Dict[str, Any]:
        """Process text for PHI detection.
        
        Args:
            text: Input clinical text
            
        Returns:
            PHI detection results
        """
        result = self.load_model("phi_detection")
        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "detections": []
            }
            
        model = result["model"]
        try:
            detections = model.detect(text)
            return {
                "success": True,
                "detections": detections,
                "count": len(detections)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "detections": []
            }
    
    def process_summarization(self, text: str, max_length: Optional[int] = None) -> Dict[str, Any]:
        """Process text for clinical summarization.
        
        Args:
            text: Input clinical text
            max_length: Maximum summary length (optional)
            
        Returns:
            Summarization results
        """
        result = self.load_model("summarization")
        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "summary": ""
            }
            
        model = result["model"]
        try:
            summary_result = model.summarize(text, max_length)
            return summary_result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": f"Summarization failed: {e}"
            }
    
    def process_diagnostics(self, text: str, summary: Optional[str] = None) -> Dict[str, Any]:
        """Process text for diagnostic suggestions.
        
        Args:
            text: Original clinical text
            summary: Clinical summary (optional)
            
        Returns:
            Diagnostic results
        """
        result = self.load_model("diagnostics")
        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "diagnostics": "PLACEHOLDER: Diagnostics model failed to load"
            }
            
        model = result["model"]
        try:
            diagnostic_result = model.generate_diagnostics(text, summary)
            return diagnostic_result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "diagnostics": f"Diagnostics failed: {e}"
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models.
        
        Returns:
            Status information for all model types
        """
        status = {}
        for model_type, config in self.model_configs.items():
            status[model_type] = {
                "enabled": config["enabled"],
                "loaded": model_type in self.loaded_models,
                "module": config["module_path"]
            }
        return status
    
    def unload_model(self, model_type: str) -> bool:
        """Unload a specific model to free memory.
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            True if successful, False otherwise
        """
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            logger.info(f"🗑️  Unloaded {model_type} model")
            return True
        return False
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        self.loaded_models.clear()
        logger.info("🗑️  All models unloaded")

def test_model_manager():
    """Test function for the model manager."""
    manager = ModelManager()
    
    print("Testing Model Manager...")
    print(f"Model Status: {manager.get_model_status()}")
    
    # Test PHI detection
    test_text = "Patient John Smith (DOB: 12/15/1980) was admitted on 2024-07-26. Contact: (555) 123-4567."
    
    print("\n1. Testing PHI Detection:")
    phi_result = manager.process_phi_detection(test_text)
    print(f"PHI Result: {phi_result}")
    
    print("\n2. Testing Summarization:")
    summary_result = manager.process_summarization(test_text)
    print(f"Summary Result: {summary_result}")
    
    print("\n3. Testing Diagnostics:")
    diag_result = manager.process_diagnostics(test_text)
    print(f"Diagnostics Result: {diag_result}")

if __name__ == "__main__":
    test_model_manager()
