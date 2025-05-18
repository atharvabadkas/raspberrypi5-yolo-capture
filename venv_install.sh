#!/bin/bash
# Python virtual environment setup for NCNN and PyTorch
# Run as normal user (no sudo): bash venv_install.sh

# Exit on error
set -e

echo "===== Setting up Python Virtual Environment for NCNN and PyTorch ====="

# Define project directory
PROJECT_DIR=~/Desktop/WMSV4AI
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create directories for models and images
mkdir -p models
mkdir -p captured_images

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv raspi

# Activate the environment
source raspi/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python packages
echo "Installing PyTorch and related packages..."
pip install torch torchvision
pip install ultralytics
pip install opencv-python-headless numpy
pip install psutil
pip install onnx  # For model conversion
pip install ncnn  # NCNN Python bindings

# Create conversion script
echo "Creating model conversion script..."
cat > convert_model.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

def convert_to_onnx(model_path, output_dir):
    """Convert YOLOv8 model to ONNX format"""
    try:
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(model_path)
        
        # Export to ONNX with optimizations
        print(f"Converting {model_path} to ONNX format...")
        onnx_path = model.export(format="onnx", dynamic=True, simplify=True)
        
        # Verify output path
        if not os.path.exists(onnx_path):
            onnx_path = Path(output_dir) / f"{Path(model_path).stem}.onnx"
            if not os.path.exists(onnx_path):
                print(f"❌ Failed to find exported ONNX model at {onnx_path}")
                return None
                
        print(f"✅ Model exported to ONNX: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ Failed to convert to ONNX: {e}")
        return None

def convert_onnx_to_ncnn(onnx_path, output_dir):
    """Convert ONNX model to NCNN format using onnx2ncnn"""
    try:
        # Get base filename without extension
        base_name = Path(onnx_path).stem
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output paths
        param_path = Path(output_dir) / f"{base_name}.param"
        bin_path = Path(output_dir) / f"{base_name}.bin"
        
        # Run onnx2ncnn command
        print(f"Converting {onnx_path} to NCNN format...")
        cmd = [
            "onnx2ncnn",
            str(onnx_path),
            str(param_path),
            str(bin_path)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ onnx2ncnn failed: {result.stderr}")
            print(f"Command output: {result.stdout}")
            return None
            
        # Verify the files were created
        if param_path.exists() and bin_path.exists():
            print(f"✅ NCNN model created: {param_path} and {bin_path}")
            return param_path, bin_path
        else:
            print("❌ NCNN conversion failed - output files not found")
            return None
    except Exception as e:
        print(f"❌ Error converting to NCNN: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv8 model to NCNN format")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", 
                        help="Path to the YOLOv8 model file")
    parser.add_argument("--output-dir", type=str, default="models", 
                        help="Directory to save the converted model")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please specify a valid model path or run main.py first to download the model.")
        return 1
    
    print(f"Converting model: {model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Convert to ONNX format
    onnx_path = convert_to_onnx(model_path, args.output_dir)
    if not onnx_path:
        return 1
    
    # Step 2: Convert with onnx2ncnn
    result = convert_onnx_to_ncnn(onnx_path, args.output_dir)
    
    if not result:
        print("❌ Model conversion failed. The model will run in PyTorch mode, which is slower.")
        return 1
    
    print("\n==== Conversion Summary ====")
    print(f"✅ Original model: {model_path}")
    print(f"✅ ONNX model: {onnx_path}")
    print(f"✅ NCNN model: {result[0]} and {result[1]}")
    print("\nThe model has been successfully converted to NCNN format.")
    print("You can now use these files with your application for better performance.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Create run script for convenience
echo "Creating convenience run script..."
cat > run.sh << 'EOF'
#!/bin/bash
cd ~/Desktop/WMSV4AI
source raspi/bin/activate
python3 "$@"
EOF

# Make scripts executable
chmod +x convert_model.py
chmod +x run.sh

# Create model update script
echo "Creating model update script..."
cat > update_model.py << 'EOF'
#!/usr/bin/env python3
import re
import os

def update_model_file():
    # Check if model.py exists
    if not os.path.exists('model.py'):
        print("❌ model.py not found. Please run in the correct directory.")
        return False

    with open('model.py', 'r') as f:
        content = f.read()
    
    # Make a backup
    with open('model.py.backup', 'w') as f:
        f.write(content)
    
    # Add better error handling for NCNN inference
    improved_ncnn_error_handling = """
            # If inference fails, try to fall back to PyTorch
            print(f"Error in NCNN inference: {e}")
            if hasattr(self, 'model_fallback') and self.model_fallback:
                print("Falling back to PyTorch model for this frame")
                try:
                    return self.model_fallback(frame)
                except Exception as e2:
                    print(f"PyTorch fallback also failed: {e2}")
            return []
    """
    
    # Find the NCNN inference method's exception handler and improve it
    pattern = r"(    except Exception as e:\s+print\(f\"Error in NCNN inference: \{e\}\"\).*?return \[\])"
    content = re.sub(pattern, 
                    "    except Exception as e:" + improved_ncnn_error_handling, 
                    content, 
                    flags=re.DOTALL)
    
    # Add capability to store a fallback PyTorch model
    load_model_update = """
        # Store PyTorch model as fallback
        if os.path.exists(self.model_path) and not self.using_ncnn:
            self.model_fallback = self.model
        else:
            self.model_fallback = None
    """
    
    # Add the fallback storage after PyTorch model loading
    pattern = r"(            print\(\"PyTorch model loaded successfully\"\)\s+self\.using_ncnn = False\s+return True)"
    content = re.sub(pattern, r"\1\n" + load_model_update, content)
    
    # Write the updated content back
    with open('model.py', 'w') as f:
        f.write(content)
    
    print("✅ model.py updated with improved NCNN error handling and fallback mechanism")
    return True

if __name__ == "__main__":
    update_model_file()
EOF

chmod +x update_model.py

# Deactivate virtual environment
deactivate

echo -e "\n===== Python Environment Setup Complete ====="
echo "To use the environment:"
echo "1. Activate: source ~/Desktop/WMSV4AI/raspi/bin/activate"
echo "2. Or use the run script: ./run.sh [script_name]"
echo ""
echo "Next steps:"
echo "1. Download the model: ./run.sh main.py (Ctrl+C after it starts downloading)"
echo "2. Update the model code: ./run.sh update_model.py"
echo "3. Convert the model: ./run.sh convert_model.py"
echo "4. Run verification: ./run.sh verification.py"
echo "5. Run the main application: ./run.sh main.py"