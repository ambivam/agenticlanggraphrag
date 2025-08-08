from ingest.file_handlers import process_powerpoint
import os

# Test PowerPoint processing
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
pptx_files = [f for f in os.listdir(data_dir) if f.endswith('.pptx')]

print(f"Found {len(pptx_files)} PowerPoint files:")
for pptx_file in pptx_files:
    file_path = os.path.join(data_dir, pptx_file)
    print(f"\nProcessing: {pptx_file}")
    try:
        docs = process_powerpoint(file_path)
        print(f"Extracted {len(docs)} chunks")
        if docs:
            print("\nSample content from first chunk:")
            print(docs[0].page_content[:200])
            print("\nMetadata:", docs[0].metadata)
    except Exception as e:
        print(f"Error processing {pptx_file}: {str(e)}")
        import traceback
        print(traceback.format_exc())
