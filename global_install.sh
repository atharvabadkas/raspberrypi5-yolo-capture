# Global installation script for Raspberry Pi OS
# Run with: sudo bash global_install.sh

# Exit on error
set -e

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run as root (use sudo)."
    exit 1
fi

echo "===== Installing System Dependencies for NCNN and PyTorch ====="

# Update package repositories
echo "Updating package repositories..."
apt update
apt upgrade -y

# Install essential build tools
echo "Installing build tools and development libraries..."
apt install -y cmake build-essential git
apt install -y libopencv-dev libopencv-core-dev
apt install -y libprotobuf-dev protobuf-compiler
apt install -y python3-dev python3-pip python3-venv python3-setuptools python3-wheel
apt install -y python3-numpy

# Install Vulkan support for GPU acceleration
echo "Installing Vulkan support for GPU acceleration..."
apt install -y libvulkan-dev vulkan-tools mesa-vulkan-drivers

# Install Raspberry Pi specific packages
echo "Installing Raspberry Pi specific packages..."
apt install -y raspberrypi-kernel-headers libraspberrypi-dev

# Install camera support
echo "Installing camera support..."
apt install -y python3-picamera2

# Configure GPU memory
echo "Configuring GPU memory..."
if grep -q "^gpu_mem=" /boot/config.txt; then
    sed -i 's/^gpu_mem=.*/gpu_mem=128/' /boot/config.txt
    echo "Updated GPU memory allocation to 128MB"
else
    echo "gpu_mem=128" >> /boot/config.txt
    echo "Added GPU memory allocation (128MB)"
fi

# Build and install NCNN tools
echo "Building NCNN tools (onnx2ncnn)..."
WORK_DIR=$(pwd)
mkdir -p ~/ncnn_build
cd ~/ncnn_build

# Check if NCNN already cloned
if [ ! -d "ncnn" ]; then
    git clone https://github.com/Tencent/ncnn.git
    cd ncnn
    git checkout 20221128  # Use a stable version
    git submodule update --init --recursive  # Initialize all submodules
else
    cd ncnn
    git fetch
    git checkout 20221128
    git submodule update --init --recursive  # Ensure submodules are up to date
fi

# Build NCNN with minimal options
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_PYTHON=OFF -DNCNN_BUILD_TOOLS=ON ..

# Build only needed tools
echo "Building onnx2ncnn tool..."
cd tools/onnx
make -j4

# Install onnx2ncnn to system path
echo "Installing onnx2ncnn to system path..."
cp onnx2ncnn /usr/local/bin/
chmod +x /usr/local/bin/onnx2ncnn

# Return to original directory
cd "$WORK_DIR"

# Configure camera
echo "Enabling camera interface..."
if command -v raspi-config > /dev/null; then
    # Use raspi-config nonint command to enable camera
    raspi-config nonint do_camera 0
    echo "Camera interface enabled through raspi-config"
else
    echo "WARNING: raspi-config not found. Please enable camera manually:"
    echo "sudo raspi-config > Interface Options > Camera > Enable"
fi

echo -e "\n===== Global Installation Complete ====="
echo "A system reboot is recommended to apply all changes."
echo "After reboot, run the Python environment setup script:"
echo "bash venv_install.sh"
echo ""
echo "To reboot now, run: sudo reboot"