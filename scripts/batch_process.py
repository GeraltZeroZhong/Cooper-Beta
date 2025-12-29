import os
import sys
import glob
import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 将项目根目录加入路径，确保能 import beta_cooper
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from beta_cooper.parser import ProteinParser
from beta_cooper.validator import BarrelValidator
from beta_cooper.geometry import BarrelGeometry

def process_single_structure(pdb_path):
    """
    Worker function for a single PDB file.
    Returns a dictionary row for the DataFrame.
    """
    filename = os.path.basename(pdb_path)
    stem = Path(pdb_path).stem
    
    # Base Result
    result = {
        "filename": filename,
        "id": stem,
        "status": "UNKNOWN",
        "confidence": 0.0,
        "n_strands": np.nan,
        "shear_S": np.nan,
        "radius": np.nan,
        "tilt": np.nan,
        "height": np.nan,
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    try:
        # --- 1. Validate ---
        # Note: Validator internal parser is fast/robust for geometry checks
        validator = BarrelValidator(pdb_path)
        v_res = validator.validate()
        
        result.update({
            "status": v_res['status'],
            "confidence": v_res['confidence'],
            "issue": v_res['issue'],
            # Flatten metrics into columns
            **v_res['metrics']
        })
        
        # If not a valid barrel, stop here
        if not v_res['is_valid']:
            result["processing_time"] = time.time() - start_time
            return result

        # --- 2. Geometry (Only for Valid Barrels) ---
        # Need strict parser for coordinates
        parser = ProteinParser(pdb_path)
        dssp_dict = parser.run_dssp()
        chain_a = parser.get_chain('A') # Assuming Chain A for now
        
        beta_segments = []
        current_seg = []
        last_resseq = -999
        
        if chain_a:
            for res in chain_a:
                key = ('A', res.id)
                if key not in dssp_dict: continue
                
                ss = dssp_dict[key][2]
                resseq = res.id[1]
                
                # Extraction Logic (same as main.py)
                if abs(resseq - last_resseq) > 4:
                    if len(current_seg) >= 3: beta_segments.append(np.array(current_seg))
                    current_seg = []
                
                if ss == 'E' and 'CA' in res:
                    current_seg.append(res['CA'].get_coord())
                    last_resseq = resseq
            
            if len(current_seg) >= 3: beta_segments.append(np.array(current_seg))

        if not beta_segments:
            result["status"] = "FAIL_NO_BETA"
            result["processing_time"] = time.time() - start_time
            return result

        # Run Geometry
        all_coords = np.vstack(beta_segments)
        geo = BarrelGeometry(segments=beta_segments, all_coords=all_coords)
        params = geo.get_summary()
        
        # Update with Physics
        result.update({
            "n_strands": params['n_strands'],
            "shear_S": params['shear_S'],
            "radius": params['radius'],
            "tilt": params['tilt_angle'],
            "height": params['height']
        })
        
    except Exception as e:
        result["status"] = "CRASH"
        result["issue"] = str(e)
        
    result["processing_time"] = time.time() - start_time
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Beta-Cooper Batch Harvester")
    parser.add_argument("--input", "-i", required=True, help="Folder containing PDB/CIF files")
    parser.add_argument("--output", "-o", default="barrel_census.csv", help="Output CSV file")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of CPU cores")
    args = parser.parse_args()

    input_dir = args.input
    output_file = args.output
    
    # 1. Gather Files
    extensions = ['*.pdb', '*.cif', '*.ent']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not files:
        print(f"No structures found in {input_dir}")
        return

    print(f"🚀 Starting Harvest on {len(files)} structures using {args.workers} cores...")
    
    results = []
    
    # 2. Parallel Processing
    # Using ProcessPoolExecutor for true parallelism (bypassing GIL)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_structure, f): f for f in files}
        
        # Progress Bar logic (Simple text based)
        total = len(files)
        completed = 0
        
        for future in as_completed(future_to_file):
            data = future.result()
            results.append(data)
            completed += 1
            
            # Print progress every 10 files or 10%
            if completed % 10 == 0 or completed == total:
                percent = (completed / total) * 100
                print(f"[{percent:.1f}%] Processed {completed}/{total} - Last: {data['filename']} ({data['status']})")

    # 3. Save Report
    df = pd.DataFrame(results)
    
    # Organize Columns (Key metrics first)
    cols = ['id', 'status', 'confidence', 'n_strands', 'shear_S', 'radius', 'tilt', 'issue']
    # Add remaining columns
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]
    
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Done! Census saved to: {output_file}")
    print("\nSummary:")
    print(df['status'].value_counts())
    
    # Quick Scientific Insight
    valid_df = df[df['status'] == 'OK']
    if not valid_df.empty:
        print(f"\n[Scientific Preview]")
        print(f"Total Valid Barrels: {len(valid_df)}")
        print(f"Avg Strand Count:    {valid_df['n_strands'].mean():.1f}")
        print(f"Avg Radius:          {valid_df['radius'].mean():.1f} A")

if __name__ == "__main__":
    main()