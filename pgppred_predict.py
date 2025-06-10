#!/usr/bin/env python3

!pip install tensorflow>=2.8.0
!pip install keras>=2.8.0
!pip install numpy>=1.21.0
!pip install scikit-learn>=1.0.0

import os
import sys
import csv
import argparse
import numpy as np
from keras.models import load_model
from keras.utils import Sequence
import glob

class SequenceGenerator(Sequence):
    def __init__(self, sequences, batch_size=32, max_length=500):
        self.sequences = sequences
        self.batch_size = batch_size
        self.max_length = max_length
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_index = {aa: idx for idx, aa in enumerate(self.amino_acids)}

    def one_hot_encode(self, seq):
        one_hot = np.zeros((self.max_length, len(self.amino_acids)), dtype=int)
        for i, aa in enumerate(seq):
            if aa in self.aa_to_index and i < self.max_length:
                one_hot[i, self.aa_to_index[aa]] = 1
        return one_hot

    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_sequences = self.sequences[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.array([self.one_hot_encode(seq) for seq in batch_sequences])
        return X

def load_fasta_sequences(file_path):
    """Load sequences from FASTA file"""
    sequences = []
    seq_headers = []
    
    try:
        with open(file_path, 'r') as file:
            sequence = ''
            header = ''
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if sequence:
                        sequences.append(sequence)
                        seq_headers.append(header)
                    header = line[1:]  # Remove '>' character
                    sequence = ''
                else:
                    sequence += line
            if sequence:
                sequences.append(sequence)
                seq_headers.append(header)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return [], []
    
    return sequences, seq_headers

def save_predictions_csv(sequences, headers, y_pred_labels, y_pred_probs, output_file, class_labels):
    """Save predictions to CSV file"""
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Header', 'Sequence', 'Predicted_Class', 'Confidence'])
        
        for header, seq, pred_idx, prob in zip(headers, sequences, y_pred_labels, y_pred_probs):
            predicted_class = class_labels[pred_idx]
            confidence = float(max(prob))
            writer.writerow([header, seq, predicted_class, confidence])

def save_sequences_by_class(sequences, headers, y_pred_labels, class_labels, output_dir, input_filename):
    """Save sequences grouped by predicted class into separate FASTA files"""
    class_to_seqs = {}
    
    for header, seq, pred_idx in zip(headers, sequences, y_pred_labels):
        class_name = class_labels[pred_idx]
        if class_name not in class_to_seqs:
            class_to_seqs[class_name] = []
        class_to_seqs[class_name].append((header, seq))
    
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    for class_name, seq_list in class_to_seqs.items():
        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        output_file = os.path.join(output_dir, f"{base_name}_{safe_class_name}.fasta")
        
        with open(output_file, 'w') as f:
            for header, seq in seq_list:
                f.write(f">{header}\n{seq}\n")
        
        print(f"Saved {len(seq_list)} sequences to {output_file}")

def predict_sequences(model_path, input_files, output_dir, batch_size=32, max_length=500):
    """Main prediction function"""
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return False
    
    # Load the pre-trained model
    try:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
    # Define class labels based on your training data
    class_labels = {0: 'Non-PGP', 1: 'PGP'}  # Adjust based on your training labels
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each input file
    for input_file in input_files:
        print(f"\nProcessing file: {input_file}")
        
        # Load sequences
        sequences, headers = load_fasta_sequences(input_file)
        
        if not sequences:
            print(f"No sequences found in {input_file}, skipping.")
            continue
        
        print(f"Found {len(sequences)} sequences")
        
        # Create data generator
        pred_gen = SequenceGenerator(sequences, batch_size=batch_size, max_length=max_length)
        
        # Make predictions
        print("Making predictions...")
        predictions_prob = model.predict(pred_gen, verbose=1)
        predictions = np.argmax(predictions_prob, axis=1)
        
        # Print summary
        print(f"\nPrediction Summary for {input_file}:")
        class_counts = {}
        for pred_idx in predictions:
            class_name = class_labels[pred_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} sequences")
        
        # Save results
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Save CSV with all predictions
        csv_output = os.path.join(output_dir, f"{base_name}_predictions.csv")
        save_predictions_csv(sequences, headers, predictions, predictions_prob, csv_output, class_labels)
        print(f"Detailed predictions saved to: {csv_output}")
        
        # Save sequences grouped by predicted class
        save_sequences_by_class(sequences, headers, predictions, class_labels, output_dir, input_file)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='PGPpred: Protein Group Prediction Tool')
    parser.add_argument('-m', '--model', required=True, help='Path to the trained model file (.h5)')
    parser.add_argument('-i', '--input', nargs='+', help='Input FASTA file(s) for prediction')
    parser.add_argument('-d', '--directory', help='Directory containing FASTA files (alternative to -i)')
    parser.add_argument('-o', '--output', default='predictions', help='Output directory (default: predictions)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for prediction (default: 32)')
    parser.add_argument('--max_length', type=int, default=500, help='Maximum sequence length (default: 500)')
    
    args = parser.parse_args()
    
    # Determine input files
    input_files = []
    
    if args.input:
        input_files = args.input
    elif args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory {args.directory} does not exist.")
            return
        
        # Find FASTA files in directory
        extensions = ['*.fasta', '*.fa', '*.txt', '*.FASTA', '*.FA', '*.TXT']
        for ext in extensions:
            input_files.extend(glob.glob(os.path.join(args.directory, ext)))
    else:
        # Default: look for FASTA files in current directory
        extensions = ['*.fasta', '*.fa', '*.txt', '*.FASTA', '*.FA', '*.TXT']
        for ext in extensions:
            input_files.extend(glob.glob(ext))
    
    if not input_files:
        print("No FASTA files found. Please specify input files or ensure FASTA files exist in the target directory.")
        return
    
    print(f"Found {len(input_files)} FASTA files for prediction:")
    for f in input_files:
        print(f"  - {f}")
    
    # Run predictions
    success = predict_sequences(args.model, input_files, args.output, args.batch_size, args.max_length)
    
    if success:
        print(f"\nPrediction completed! Results saved in '{args.output}' directory.")
    else:
        print("\nPrediction failed!")

if __name__ == "__main__":
    main()
