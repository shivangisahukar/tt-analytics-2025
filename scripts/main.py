import os
from scripts.mapping.header_mapping import run_mapping_pipeline

def main():
    print("="*40)
    print("STARTING TABLE TENNIS DATA PIPELINE")
    print("="*40)
    
    # 1. Define paths
    raw_folder = "data/raw"
    processed_folder = "data/processed"
    
    # 2. Safety Check: Ensure directories exist
    if not os.path.exists(raw_folder):
        print(f"ERROR: Raw data folder not found at {raw_folder}")
        return
    
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
        print(f"Created directory: {processed_folder}")

    # 3. Trigger the CSV Mapping Pipeline
    # This calls the function we wrote in header_mapping.py
    try:
        run_mapping_pipeline()
        print("\nSUCCESS: Master Long-Format Dataset has been generated.")
        print(f"Location: {processed_folder}/master_long_dataset.csv")
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")

    print("="*40)

if __name__ == "__main__":
    main()