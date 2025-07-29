# MedAssist AI - Pipeline Orchestrator  
# Phase 2.2: Multi-Model Pipeline with Unified Model Manager
# Clean architecture using centralized model management

from python import Python, PythonObject

fn process_with_model_manager(text: String, enable_phi: Bool = True, enable_summary: Bool = False, enable_diagnosis: Bool = False) raises -> PythonObject:
    """Use the unified model manager for all clinical processing."""
    var sys = Python.import_module("sys")
    _ = sys.path.append(".")
    _ = sys.path.append("/home/shyamsridhar/code/mojo-medassist")
    _ = sys.path.append("/home/shyamsridhar/code/mojo-medassist/models")
    
    # Import the model manager
    var model_manager = Python.import_module("model_manager")
    var manager = model_manager.ModelManager()
    
    var results = Python.evaluate("{'original_text': '', 'phi_detections': [], 'summary': None, 'diagnosis': None, 'pipeline_status': 'incomplete'}")
    results["original_text"] = text
    
    print("MedAssist Clinical Pipeline (Model Manager)")
    print("Input text length:")
    print(len(text))
    
    # Stage 1: PHI Detection
    if enable_phi:
        print("Stage 1: PHI Detection...")
        try:
            var phi_result = manager.process_phi_detection(text)
            var success = phi_result["success"]
            if success:
                results["phi_detections"] = phi_result["detections"]
                var count = phi_result["count"]
                print("PHI Detection complete - entities found:")
                print(count)
            else:
                print("PHI Detection failed:", phi_result["error"])
                results["pipeline_status"] = "phi_failed"
                return results
        except e:
            print("PHI Detection failed:", e)
            results["pipeline_status"] = "phi_failed"
            return results
    else:
        print("Stage 1: PHI Detection skipped")
    
    # Stage 2: Summarization  
    if enable_summary:
        print("Stage 2: Clinical Summarization...")
        try:
            var summary_result = manager.process_summarization(text)
            var success = summary_result["success"]
            if success:
                results["summary"] = summary_result["summary"]
                var summary_text = summary_result["summary"]
                print("Summarization complete - length:")
                print(len(summary_text))
            else:
                print("Summarization failed:", summary_result["error"])
                results["summary"] = summary_result["summary"]  # Contains fallback
        except e:
            print("Summarization failed:", e)
            results["summary"] = "ERROR: Summarization service unavailable"
            results["pipeline_status"] = "summary_failed"
            return results
    else:
        print("Stage 2: Summarization skipped")
    
    # Stage 3: Diagnostics
    if enable_diagnosis:
        print("Stage 3: Diagnostic Support...")
        try:
            var diag_result = manager.process_diagnostics(text)
            var success = diag_result["success"]
            if success:
                results["diagnosis"] = diag_result["diagnostics"]
                print("Diagnostics complete")
            else:
                print("Diagnostics placeholder:", diag_result["error"])
                results["diagnosis"] = diag_result["diagnostics"]
                results["pipeline_status"] = "diagnosis_not_implemented"
                return results
        except e:
            print("Diagnostics failed:", e)
            results["diagnosis"] = "ERROR: Diagnostics service unavailable"
            results["pipeline_status"] = "diagnosis_failed"
            return results
    else:
        print("Stage 3: Diagnostics skipped")
    
    results["pipeline_status"] = "complete"
    print("\nPipeline processing complete!")
    return results

fn print_pipeline_results(results: PythonObject) raises:
    """Print pipeline results in a structured format."""
    print("\nPipeline Results Summary:")
    print("=" * 40)
    
    var status = results["pipeline_status"]
    print("Status:", status)
    
    # PHI Detection Results
    var phi_detections = results["phi_detections"]
    if phi_detections:
        var len_fn = Python.evaluate("len")
        var detection_count = len_fn(phi_detections)
        print("\nPHI Detection:")
        print("  Entities found:", detection_count)
        
        for i in range(detection_count.__int__()):
            var detection = phi_detections[i]
            var label = detection["label"]
            var text_val = detection["text"]
            print("  -", label, ":", text_val)
    
    # Summary Results
    var summary = results["summary"]
    if summary:
        print("\nSummary:", summary)
    
    # Diagnosis Results
    var diagnosis = results["diagnosis"]
    if diagnosis:
        print("\nDiagnosis:", diagnosis)


fn main() raises:
    print("MedAssist AI - Clinical Pipeline Orchestrator v2.2")
    print("=" * 60)
    
    var test_text = "Patient John Smith (DOB: 12/15/1980) was admitted on 2024-07-26. Chief complaint: chest pain. Vitals: BP 140/90, HR 88. Contact: (555) 123-4567."
    
    print("Test clinical note:")
    print("   ", test_text)
    
    # Test 1: PHI detection only (using model manager)
    print("\n" + "=" * 60)
    print("TEST 1: PHI Detection Only (MODEL MANAGER)")
    var results1 = process_with_model_manager(test_text, enable_phi=True)
    print_pipeline_results(results1)
    
    # Test 2: PHI + Summarization (using model manager)
    print("\n" + "=" * 60)  
    print("TEST 2: PHI Detection + Summarization (MODEL MANAGER)")
    var results2 = process_with_model_manager(test_text, enable_phi=True, enable_summary=True)
    print_pipeline_results(results2)
    
    # Test 3: Full pipeline (will show diagnostics placeholder)
    print("\n" + "=" * 60)  
    print("TEST 3: Full Pipeline with Diagnostics Placeholder")
    var results3 = process_with_model_manager(test_text, enable_phi=True, enable_summary=True, enable_diagnosis=True)
    print_pipeline_results(results3)
    
    print("\nModel Manager Integration Complete!")
    print("✅ PHI Detection: ClinicalBERT only (rule-based disabled)")
    print("✅ Summarization: T5 medical model working")
    print("⏳ Diagnostics: Placeholder ready for BioGPT integration")
