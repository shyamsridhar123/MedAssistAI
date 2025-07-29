"""
MedAssist AI - Clinical Diagnostics Module
Phase 2.2.3: MedGemma 4B GGUF-based Diagnostic Support

Implements CPU-optimized MedGemma diagnostics using llama.cpp with GGUF quantization.
"""

import logging
import os
import subprocess
import json
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalDiagnostics:
    """MedGemma 4B GGUF-based diagnostic support for clinical notes."""
    
    def __init__(self, model_path: str = None, llama_binary: str = None):
        """Initialize the clinical diagnostics system.
        
        Args:
            model_path: Path to the MedGemma GGUF model. If None, uses default path.
            llama_binary: Path to llama-cli binary. If None, uses default path.
        """
        self.model_path = model_path or "models/diagnostics/gguf/medgemma-4b-it-Q4_K_M.gguf"
        self.llama_binary = llama_binary or "external/llama.cpp/build/bin/llama-cli"
        self.is_initialized = False
        
        # Diagnostic generation settings
        self.max_tokens = 300
        self.temperature = 0.7
        self.top_p = 0.9
        self.threads = 4
        self.batch_size = 512
        
    def initialize(self) -> Dict[str, any]:
        """Initialize the MedGemma diagnostics system.
        
        Returns:
            Dict with initialization status and details.
        """
        try:
            logger.info("🚀 Initializing MedGemma 4B Diagnostic System")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                return {
                    "success": False,
                    "error": f"MedGemma model not found: {self.model_path}",
                    "fallback_available": True
                }
            
            # Check if llama-cli binary exists
            if not os.path.exists(self.llama_binary):
                return {
                    "success": False,
                    "error": f"llama-cli binary not found: {self.llama_binary}",
                    "fallback_available": True
                }
            
            logger.info(f"🔽 Loading MedGemma 4B from {self.model_path}")
            
            # Test model loading with a quick inference
            test_result = self._run_inference("Test", max_tokens=1)
            if not test_result["success"]:
                return {
                    "success": False,
                    "error": f"Model test failed: {test_result['error']}",
                    "fallback_available": True
                }
            
            self.is_initialized = True
            logger.info("✅ MedGemma 4B Diagnostics loaded successfully")
            
            return {
                "success": True,
                "model_type": "MedGemma 4B GGUF",
                "quantization": "Q4_K_M",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MedGemma diagnostics: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_available": True
            }
    
    def generate_diagnostics(self, clinical_text: str, summary: Optional[str] = None) -> Dict[str, any]:
        """Generate diagnostic suggestions from clinical text.
        
        Args:
            clinical_text: Input clinical text
            summary: Optional clinical summary for context
            
        Returns:
            Dict with diagnostic results and metadata
        """
        if not self.is_initialized:
            init_result = self.initialize()
            if not init_result["success"]:
                return {
                    "success": False,
                    "diagnostics": "",
                    "error": "Model not initialized",
                    "fallback_used": True,
                    "fallback_diagnostics": self._generate_fallback_diagnostics(clinical_text)
                }
        
        try:
            # Prepare diagnostic prompt
            prompt = self._create_diagnostic_prompt(clinical_text, summary)
            
            # Generate diagnostics using MedGemma
            result = self._run_inference(prompt, max_tokens=self.max_tokens)
            
            if result["success"]:
                diagnostics = result["output"].strip()
                
                # Clean and validate diagnostics
                if not diagnostics or len(diagnostics) < 10:
                    fallback_diagnostics = self._generate_fallback_diagnostics(clinical_text)
                    return {
                        "success": False,
                        "diagnostics": fallback_diagnostics,
                        "error": "Generated diagnostics too short",
                        "fallback_used": True,
                        "model_type": "Rule-based fallback"
                    }
                
                logger.info(f"✅ Diagnostics generated: {len(diagnostics)} characters")
                
                return {
                    "success": True,
                    "diagnostics": diagnostics,
                    "input_length": len(clinical_text),
                    "output_length": len(diagnostics),
                    "fallback_used": False,
                    "model_type": "MedGemma 4B"
                }
            else:
                # Fall back to rule-based diagnostics
                fallback_diagnostics = self._generate_fallback_diagnostics(clinical_text)
                return {
                    "success": False,
                    "diagnostics": fallback_diagnostics,
                    "error": result["error"],
                    "fallback_used": True,
                    "model_type": "Rule-based fallback"
                }
                
        except Exception as e:
            logger.error(f"❌ Diagnostics generation failed: {e}")
            
            # Try fallback diagnostics
            fallback_diagnostics = self._generate_fallback_diagnostics(clinical_text)
            
            return {
                "success": False,
                "diagnostics": fallback_diagnostics,
                "error": str(e),
                "fallback_used": True,
                "model_type": "Rule-based fallback"
            }
    
    def _create_diagnostic_prompt(self, clinical_text: str, summary: Optional[str] = None) -> str:
        """Create a clinical diagnostic prompt for MedGemma.
        
        Args:
            clinical_text: Original clinical text
            summary: Optional clinical summary
            
        Returns:
            Formatted prompt for diagnostic generation
        """
        base_prompt = """You are a clinical AI assistant providing diagnostic support. Based on the clinical information provided, suggest possible diagnoses and next steps. Focus on evidence-based differential diagnoses.

Clinical Information:
{clinical_text}"""
        
        if summary:
            base_prompt += f"""

Clinical Summary:
{summary}"""
        
        base_prompt += """

Provide a structured diagnostic assessment with:
1. Primary differential diagnoses
2. Key clinical findings
3. Recommended next steps or tests

Diagnostic Assessment:"""
        
        return base_prompt.format(clinical_text=clinical_text)
    
    def _run_inference(self, prompt: str, max_tokens: int = None) -> Dict[str, any]:
        """Run inference using llama-cli with MedGemma.
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with inference results
        """
        try:
            max_tokens = max_tokens or self.max_tokens
            
            # Build llama-cli command
            cmd = [
                self.llama_binary,
                "-m", self.model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "-t", str(self.threads),
                "-b", str(self.batch_size),
                "--temp", str(self.temperature),
                "--top-p", str(self.top_p),
                "--no-conversation",
                "--no-display-prompt"
            ]
            
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "success": False,
                    "error": f"llama-cli failed: {result.stderr}",
                    "output": ""
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Inference timeout (2 minutes)",
                "output": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    def _generate_fallback_diagnostics(self, clinical_text: str) -> str:
        """Generate rule-based diagnostic suggestions as fallback.
        
        Args:
            clinical_text: Input clinical text
            
        Returns:
            Basic diagnostic suggestions
        """
        text_lower = clinical_text.lower()
        
        # Common clinical patterns and associated diagnostic considerations
        diagnostic_patterns = {
            "chest pain": ["Acute coronary syndrome", "Pulmonary embolism", "Aortic dissection", "Pneumothorax"],
            "shortness of breath": ["Heart failure", "Pulmonary embolism", "Pneumonia", "Asthma exacerbation"],
            "abdominal pain": ["Appendicitis", "Cholecystitis", "Peptic ulcer", "Bowel obstruction"],
            "headache": ["Migraine", "Tension headache", "Cluster headache", "Secondary headache"],
            "fever": ["Viral syndrome", "Bacterial infection", "Urinary tract infection", "Pneumonia"],
            "fatigue": ["Anemia", "Thyroid dysfunction", "Depression", "Chronic fatigue syndrome"],
            "nausea": ["Gastroenteritis", "Medication side effect", "Pregnancy", "Migraine"]
        }
        
        # Find relevant patterns
        relevant_diagnoses = []
        for pattern, diagnoses in diagnostic_patterns.items():
            if pattern in text_lower:
                relevant_diagnoses.extend(diagnoses)
        
        if relevant_diagnoses:
            unique_diagnoses = list(set(relevant_diagnoses))
            diagnostic_text = "Diagnostic Considerations:\n"
            for i, diagnosis in enumerate(unique_diagnoses[:5], 1):  # Limit to top 5
                diagnostic_text += f"{i}. {diagnosis}\n"
            
            diagnostic_text += "\nRecommendations:\n"
            diagnostic_text += "- Complete history and physical examination\n"
            diagnostic_text += "- Review vital signs and laboratory results\n"
            diagnostic_text += "- Consider imaging studies if indicated\n"
            diagnostic_text += "- Monitor patient response to treatment\n"
            
            return diagnostic_text
        else:
            return """Diagnostic Assessment:
Clinical information requires comprehensive evaluation.

Recommendations:
- Complete detailed history and physical examination
- Review all available diagnostic studies
- Consider consultation with appropriate specialists
- Develop differential diagnosis based on clinical findings
- Implement evidence-based treatment plan"""

def test_diagnostics():
    """Test function for the clinical diagnostics."""
    diagnostics = ClinicalDiagnostics()
    
    test_text = """
    Patient presents with acute chest pain, shortness of breath, and sweating.
    Onset was sudden while at rest. Pain described as crushing, radiating to left arm.
    Vital signs: BP 160/95, HR 110, RR 24, O2 sat 92% on room air.
    EKG shows ST elevations in leads II, III, aVF.
    """
    
    print("Testing Clinical Diagnostics...")
    init_result = diagnostics.initialize()
    print(f"Initialization: {init_result}")
    
    if init_result["success"]:
        result = diagnostics.generate_diagnostics(test_text)
        print(f"Diagnostics Result: {result}")
        print(f"Diagnostics: {result['diagnostics']}")
    else:
        print("❌ Model initialization failed, testing fallback")
        result = diagnostics.generate_diagnostics(test_text)
        print(f"Fallback Result: {result}")

if __name__ == "__main__":
    test_diagnostics()
