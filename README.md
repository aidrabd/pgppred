# PGPpred: Protein Group Prediction Tool

A deep learning tool for protein sequence classification using a hybrid CNN-RNN model. This tool predicts whether protein sequences belong to PGP or Non-PGP groups.

## Features

- Predicts protein sequences from FASTA files (.fa, .fasta, .txt)
- Uses a pre-trained hybrid CNN-RNN deep learning model
- Supports batch processing of multiple files
- Outputs predictions in CSV format and grouped FASTA files
- Command-line interface for easy integration into workflows

## Installation

### Prerequisites

- Ubuntu/Linux termina
- pip3

### Quick Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pgppred.git
cd pgppred
```

2. Run the installation script:
```bash
chmod +x install.sh
./install.sh
```

3. Reload your shell configuration:
```bash
source ~/.bashrc
```

### Manual Installation

1. Install Python dependencies:
```bash
pip3 install -r requirements.txt
```

2. Make the script executable:
```bash
chmod +x pgppred_predict.py
```

### Installation using Miniconda

1. Install  Miniconda (if not installed)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
2. Activate conda base environment
```
conda init
```
Then, restart your terminal or run:
```
source ~/.bashrc
```
After that, activate the base environment with:
```
conda activate
```
Make sure you have Python specific version installed:
```
conda create -n py312 python=3.12.9
conda activate py312
python --version
```
Make sure you have specific Tensorflow, Keras, numpy, scikit-learn versions installed:
```
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
scikit-learn>=1.0.0

conda install -c conda-forge "tensorflow>=2.8.0"
conda install -c conda-forge "keras>=2.8.0"
conda install -c conda-forge "numpy>=1.21.0"
conda install -c conda-forge "scikit-learn>=1.0.0"
```

# Activate Python 3.12
```
conda activate py312
```

## Usage

### Basic Usage

Place your FASTA files in the current directory and run:

```bash
pgppred -m /path/to/pgppred.h5
```

This will predict all .fa, .fasta, and .txt files in the current directory.

### Advanced Usage

1. **Predict specific files:**
```bash
pgppred -m pgppred.h5 -i protein1.fasta protein2.fa protein3.txt
```

2. **Predict files in a specific directory:**
```bash
pgppred -m pgppred.h5 -d /path/to/fasta/files
```

3. **Specify output directory:**
```bash
pgppred -m pgppred.h5 -o my_results
```

4. **Adjust batch size for performance:**
```bash
pgppred -m pgppred.h5 -b 64
```

### Command Line Options

- `-m, --model`: Path to the trained model file (.h5) [REQUIRED]
- `-i, --input`: Input FASTA file(s) for prediction
- `-d, --directory`: Directory containing FASTA files
- `-o, --output`: Output directory (default: predictions)
- `-b, --batch_size`: Batch size for prediction (default: 32)
- `--max_length`: Maximum sequence length (default: 500)
- `-h, --help`: Show help message

## Output

The tool generates two types of output:

1. **CSV file**: Contains detailed predictions with sequence headers, sequences, predicted classes, and confidence scores
2. **Grouped FASTA files**: Sequences separated by predicted class (e.g., `input_PGP.fasta`, `input_Non-PGP.fasta`)

## Model Information

The model was trained using:
- **Training data**: PGP.fasta and Non-PGP.fasta
- **Architecture**: Hybrid CNN-RNN model
- **Input**: One-hot encoded amino acid sequences (max length: 500)
- **Classes**: PGP and Non-PGP

## Supported File Formats

- `.fasta`
- `.fa`
- `.txt`
- Case-insensitive extensions (`.FASTA`, `.FA`, `.TXT`)

## Example

```bash
# Place your FASTA files in current directory
ls *.fasta
# output: test_proteins.fasta

# Run prediction
pgppred -m pgppred.h5

# Check results
ls predictions/
# output: test_proteins_predictions.csv  test_proteins_PGP.fasta  test_proteins_Non-PGP.fasta
```

## Troubleshooting

### Common Issues

1. **Model file not found**: Ensure the path to your `.h5` model file is correct
2. **No FASTA files found**: Check that your files have supported extensions (.fa, .fasta, .txt)
3. **Memory errors**: Try reducing batch size with `-b` option
4. **Import errors**: Ensure all dependencies are installed: `pip3 install -r requirements.txt`

### Performance Tips

- Use larger batch sizes (`-b 64` or `-b 128`) for faster processing on systems with sufficient RAM
- For very large files, consider splitting them into smaller chunks

## Requirements

- tensorflow>=2.8.0
- keras>=2.8.0
- numpy>=1.21.0
- scikit-learn>=1.0.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

[Add citation information if this tool is published]

## Contact

Author: Saborni Sarker
GitHub: https://github.com/aidrabd/pgppred
