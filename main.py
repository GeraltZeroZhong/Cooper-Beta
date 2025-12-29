import sys
import os
import numpy as np
import pandas as pd
from beta_cooper.parser import ProteinParser
from beta_cooper.validator import BarrelValidator
from beta_cooper.geometry import BarrelGeometry

def run_pipeline_demo(pdb_path):
    print(f"==================================================")
    print(f"   Beta-Cooper Pipeline: {os.path.basename(pdb_path)}")
    print(f"==================================================\n")

    # --- Step 1: Parser ---
    try:
        parser = ProteinParser(pdb_path)
        print(f"[Step 1] Structure Loaded.")
    except Exception as e:
        print(f"Parser Error: {e}")
        return

    # --- Step 2: Validator ---
    # Validator runs its own internal DSSP and extraction logic
    validator = BarrelValidator(pdb_path, chain_id='A')
    v_res = validator.validate()
    
    print(f"[Step 2] Validation Status: {v_res['status']} (Confidence: {v_res['confidence']:.2f})")
    
    if not v_res['is_valid']:
        print(f"-> Reject: {v_res['issue']}")
        return

    # --- Step 3: Geometry (The Fix) ---
    print(f"\n[Step 3] Running Geometry Analysis...")
    
    # CRITICAL FIX: Extract ONLY Beta-Sheet atoms ('E') for geometry calculation.
    # Including loops dilutes the tilt angle and shear number.
    
    try:
        dssp_dict = parser.run_dssp() # Ensure DSSP is run
        chain_a = parser.get_chain('A')
        
        beta_segments = []
        current_segment = []
        last_resseq = -999
        
        # Iterate residues to extract 'E' segments
        # Note: parser.get_chain returns Bio.PDB.Chain object
        for res in chain_a:
            # Check if residue is in DSSP dict
            # Key format usually (chain, res_id)
            # Biopython DSSP keys vary by version, robust check:
            key = ('A', res.id)
            if key not in dssp_dict:
                continue
                
            ss = dssp_dict[key][2] # SS code is index 2
            resseq = res.id[1]
            
            # Jump detection (Chain break)
            if abs(resseq - last_resseq) > 4:
                if len(current_segment) >= 3:
                    beta_segments.append(np.array(current_segment))
                current_segment = []
            
            if ss == 'E' and 'CA' in res:
                current_segment.append(res['CA'].get_coord())
                last_resseq = resseq
                
        # Flush last
        if len(current_segment) >= 3:
            beta_segments.append(np.array(current_segment))
            
        if not beta_segments:
            print("Error: No beta segments found for geometry.")
            return
            
        # Flatten for the 'all_coords' arg, but pass segments for topology
        beta_coords_flat = np.vstack(beta_segments)
        
    except Exception as e:
        print(f"Geometry Prep Error: {e}")
        # Fallback to crude CA if DSSP fails here (unlikely if Validator passed)
        return

    # Pass the CLEAN beta segments to Geometry
    geo = BarrelGeometry(segments=beta_segments, all_coords=beta_coords_flat)
    
    params = geo.get_summary()
    
    print(f"\n--- Physical Barrel Properties ---")
    print(f"Strand Count (n):   {params['n_strands']}")
    print(f"Shear Number (S):   {params['shear_S']} (Raw: {params['shear_S_raw']})")
    print(f"Tilt Angle:         {params['tilt_angle']} deg")
    print(f"Radius:             {params['radius']} A")
    print(f"Height:             {params['height']} A")
    
    # Validation Logic for OmpA
    # n=8 is strict. S can be 10 +/- 2 due to structure flexibility.
    if params['n_strands'] == 8 and abs(params['shear_S'] - 10) <= 2:
        print("\n✅ SUCCESS: Detected correct OmpA topology!")
    else:
        print(f"\n⚠️ WARNING: Topology mismatch (Expected n=8, S=10)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdb_file>")
    else:
        run_pipeline_demo(sys.argv[1])