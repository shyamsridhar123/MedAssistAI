"""
MedAssist AI - Mojo Pipeline Bridge
Phase 1.2: Text Processing Bridge - COMPLETE

Handles Mojo-Python interoperability for PHI detection pipeline.
"""
from python import Python, PythonObject

fn detect_phi_mojo(text: String) raises -> PythonObject:
    """Mojo wrapper for PHI detection with proper error handling."""
    var sys = Python.import_module("sys")
    _ = sys.path.append(".")
    _ = sys.path.append("/home/shyamsridhar/code/mojo-medassist")
    
    # Import hybrid detector
    var hybrid_detector = Python.import_module("src.phi_detection.hybrid_detector")
    
    # Create detector instance
    var detector = hybrid_detector.HybridPHIDetector()
    _ = detector.initialize()
    
    # Detect PHI and return results
    var detections = detector.detect(text)
    return detections

fn print_detection_results(detections: PythonObject) raises:
    """Print detection results in a structured format."""
    var len_fn = Python.evaluate("len")
    var detection_count = len_fn(detections)
    print("\n🎯 Found", detection_count, "PHI entities:")
    
    for i in range(detection_count.__int__()):
        var detection = detections[i]
        var label = detection["label"]
        var text_val = detection["text"] 
        var confidence = detection["confidence"]
        var method = detection["method"]
        
        var idx = i + 1
        print("  ", idx, ".", label, ":", "'", text_val, "'", "(confidence:", confidence, ", method:", method, ")")

fn main():
    print("🚀 MedAssist AI - Mojo Pipeline v1.2")
    print("=" * 50)
    
    # Test 1: Basic Python import
    try:
        var sys = Python.import_module("sys")
        print("✅ Python import successful")
        print("Python version:", sys.version)
    except:
        print("❌ Python import failed")
        return
    
    # Test 2: Clinical text processing
    var test_text = "Patient John Smith (DOB: 12/15/1980) was admitted on 2024-07-26. Contact: (555) 123-4567. Email: john.smith@email.com SSN: 123-45-6789"
    print("📝 Test input:", test_text)
    
    try:
        print("\n🔄 Processing with Mojo-accelerated pipeline...")
        var detections = detect_phi_mojo(test_text)
        print("✅ PHI detection completed successfully!")
        
        # Print results with structured format
        print_detection_results(detections)
        
        print("\n✅ Mojo → Python → Mojo data flow working!")
        print("🎯 Chunk 1.2 COMPLETE: Text Processing Bridge successful!")
        
    except e:
        print("❌ Pipeline failed:", e)
        return
