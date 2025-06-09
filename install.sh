#!/bin/bash

# PGPpred Installation Script

echo "======================================"
echo "Installing PGPpred Tool"
echo "======================================"

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install Python3 first."
    exit 1
fi

echo "Python3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip3..."
    sudo apt update
    sudo apt install -y python3-pip
fi

echo "Installing required Python packages..."

# Install required packages
pip3 install --user tensorflow keras numpy scikit-learn

# Make the prediction script executable
chmod +x pgppred_predict.py

# Create symbolic link to make it accessible from anywhere (optional)
echo "Creating symbolic link..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/.local/bin

# Create a wrapper script in ~/.local/bin
cat > ~/.local/bin/pgppred << EOF
#!/bin/bash
python3 "$SCRIPT_DIR/pgppred_predict.py" "\$@"
EOF

chmod +x ~/.local/bin/pgppred

# Add ~/.local/bin to PATH if not already present
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "Added ~/.local/bin to PATH in ~/.bashrc"
    echo "Please run 'source ~/.bashrc' or restart your terminal"
fi

echo "======================================"
echo "Installation completed!"
echo "======================================"
echo ""
echo "Usage examples:"
echo "1. Predict files in current directory:"
echo "   pgppred -m path/to/pgppred.h5"
echo ""
echo "2. Predict specific files:"
echo "   pgppred -m path/to/pgppred.h5 -i file1.fasta file2.fa"
echo ""
echo "3. Predict files in a directory:"
echo "   pgppred -m path/to/pgppred.h5 -d /path/to/fasta/files"
echo ""
echo "4. Specify custom output directory:"
echo "   pgppred -m path/to/pgppred.h5 -o my_results"
echo ""
echo "For help: pgppred -h"