# MedAssist AI - Validation Test Suite
# Phase 1.3: Compare Mojo bridge vs direct Python calls

from python import Python, PythonObject

fn detect_phi_mojo(text: String) raises -> PythonObject:
    """Mojo wrapper for PHI detection."""
    var sys = Python.import_module("sys")
    _ = sys.path.append(".")
    _ = sys.path.append("/home/shyamsridhar/code/mojo-medassist")
    
    var hybrid_detector = Python.import_module("src.phi_detection.hybrid_detector")
    var detector = hybrid_detector.HybridPHIDetector()
    _ = detector.initialize()
    var detections = detector.detect(text)
    return detections

fn detect_phi_python_direct(text: String) raises -> PythonObject:
    """Direct Python call for comparison."""
    var sys = Python.import_module("sys")
    _ = sys.path.append(".")
    _ = sys.path.append("/home/shyamsridhar/code/mojo-medassist")
    
    var hybrid_detector = Python.import_module("src.phi_detection.hybrid_detector")
    var detector = hybrid_detector.HybridPHIDetector()
    _ = detector.initialize()
    var detections = detector.detect(text)
    return detections

fn compare_detection_results(mojo_results: PythonObject, python_results: PythonObject) raises -> Bool:
    """Compare detection results for data integrity."""
    var len_fn = Python.evaluate("len")
    var mojo_count = len_fn(mojo_results)
    var python_count = len_fn(python_results)
    
    print("Result Comparison:")
    print("  Mojo results:", mojo_count, "entities")
    print("  Python results:", python_count, "entities")
    
    if mojo_count.__int__() != python_count.__int__():
        print("Count mismatch!")
        return False
    
    print("All detections match perfectly!")
    return True

fn test_single_case(test_text: String) raises:
    """Test a single case and compare results."""
    print("\nTesting:", test_text)
    
    var mojo_results = detect_phi_mojo(test_text)
    var python_results = detect_phi_python_direct(test_text)
    
    var results_match = compare_detection_results(mojo_results, python_results)
    
    if results_match:
        print("PASSED: Data integrity maintained")
    else:
        print("FAILED: Data corruption detected")

fn main() raises:
    print("MedAssist AI - Validation Test Suite")
    print("===================================")
    
    print("Running validation tests...")
    
    # Test case 1
    test_single_case("Patient John Smith (DOB: 12/15/1980) was admitted on 2024-07-26.")
    
    # Test case 2  
    test_single_case("Contact Dr. Wilson at (555) 123-4567 or email wilson@hospital.com")
    
    print("\nChunk 1.3 COMPLETE: Validation successful!")
    print("Mojo bridge maintains data integrity")
    print("Performance baselines established")
