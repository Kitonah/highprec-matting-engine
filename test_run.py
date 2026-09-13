from PIL import Image
from pipeline import HighPrecisionExtractionEngine
import os

if __name__ == "__main__":
    # Place any image named 'input.jpg' in the project folder
    test_image_path = "input.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Please drop a test image named '{test_image_path}' in this folder to test.")
    else:
        print("Loading image...")
        img = Image.open(test_image_path)
        
        engine = HighPrecisionExtractionEngine()
        print("Processing...")
        result = engine.execute(img)
        
        result.save("output_cutout.png")
        print("Success! Cutout saved as 'output_cutout.png'")