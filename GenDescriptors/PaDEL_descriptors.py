import numpy as np
import pandas as pd
import logging
import argparse
import time
import os
import subprocess
import sys
from padelpy import from_smiles

class PaDELDescriptorCalculator:
    """Class to calculate PaDEL descriptors for molecules with optional CPU usage control."""
    
    def __init__(self, input_file, output_file, chunk_size=16, threads=8, delay=0, use_cpulimit=False, cpulimit=70):
        self.input_file = input_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.threads = threads  # Limit the number of threads used by PaDEL
        self.delay = delay  # Delay between chunk processing (seconds)
        self.use_cpulimit = use_cpulimit  # Whether to use cpulimit
        self.cpulimit = cpulimit  # CPU limit percentage (for cpulimit)
        self.write_header = not os.path.exists(self.output_file)  # Only write header if file does not exist
        logging.info(f"Initialized PaDELDescriptorCalculator with input file: {self.input_file}")

    def load_data(self):
        """Load the dataset with canonical SMILES and ChemBL IDs."""
        try:
            logging.info(f"Loading data from {self.input_file}")
            df = pd.read_csv(self.input_file)
            if 'canonical_smiles' not in df.columns:
                raise ValueError(f"Input file must contain 'canonical_smiles' column.")
            return df[['canonical_smiles']]
        except Exception as e:
            logging.error(f"Error loading input data: {e}")
            raise

    def filter_processed_smiles(self, df_selection):
        """Remove already processed SMILES strings based on existing output file."""
        if os.path.exists(self.output_file):
            logging.info(f"Reading existing output file {self.output_file} to skip processed molecules.")
            processed_df = pd.read_csv(self.output_file)
            processed_smiles = processed_df.index.tolist()

            # Filter out already processed molecules
            df_selection = df_selection[~df_selection.index.isin(processed_smiles)]
            logging.info(f"{len(processed_smiles)} molecules already processed. {len(df_selection)} remaining.")
        
        return df_selection

    def calculate_descriptors(self, smiles_list):
        """Calculate PaDEL descriptors in chunks to manage memory and processing."""
        try:
            # Split SMILES into chunks for processing
            for i, chunk in enumerate(self.chunks(smiles_list, self.chunk_size)):
                try:
                    logging.info(f"Processing chunk {i + 1} of size {len(chunk)}")
                    # Compute descriptors for the current chunk
                    descriptors = from_smiles(chunk, fingerprints=True, threads=self.threads)

                    # Save the current chunk of descriptors to the file
                    self.save_descriptors(descriptors)
                    
                    # Add a delay between processing chunks if specified
                    if self.delay > 0:
                        logging.info(f"Sleeping for {self.delay} seconds to throttle CPU usage.")
                        time.sleep(self.delay)

                except RuntimeError as e:
                    logging.error(f"Error processing chunk {i + 1}: {e}")
                    # Generate NaN values for each molecule in the chunk if error occurs
                    num_descriptors = len(descriptors[0]) if descriptors else 1000  # Estimate number of descriptors
                    nan_descriptor = {f'descriptor_{i}': np.nan for i in range(num_descriptors)}
                    descriptors = [nan_descriptor] * len(chunk)  # Fill with NaNs for each molecule in the chunk
                    
                    # Save NaN descriptors to the file
                    self.save_descriptors(descriptors)

            logging.info("Descriptor calculation completed.")

        except Exception as e:
            logging.error(f"Error during descriptor calculation: {e}")
            raise

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def save_descriptors(self, descriptors):
        """Save the PaDEL descriptors to the output file in append mode."""
        try:
            # Convert list of dictionaries (descriptors) to a DataFrame
            df_descriptors = pd.DataFrame(descriptors)

            # Append to the CSV file, writing the header only on the first write
            mode = 'a'  # Append mode
            header = self.write_header  # Write header only if it's the first chunk
            df_descriptors.to_csv(self.output_file, mode=mode, index=False, header=header)

            # After the first write, ensure headers are not written again
            if self.write_header:
                self.write_header = False
            logging.info(f"Saved chunk of descriptors to {self.output_file}")

        except Exception as e:
            logging.error(f"Error saving descriptors: {e}")
            raise

    def run_with_cpulimit(self):
        """Run the script with CPU limit using the cpulimit tool."""
        logging.info(f"Running script with CPU limit of {self.cpulimit}% using cpulimit.")
        cmd = [
            "cpulimit", "--limit", str(self.cpulimit),
            "--", "python", sys.argv[0], 
            self.input_file, self.output_file,
            "--chunk_size", str(self.chunk_size),
            "--threads", str(self.threads),
            "--delay", str(self.delay)
        ]
        logging.info(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd)

def main(input_file, output_file, chunk_size, threads, delay, use_cpulimit, cpulimit):
    logging.basicConfig(level=logging.INFO)

    # Initialize the PaDELDescriptorCalculator class
    calculator = PaDELDescriptorCalculator(input_file, output_file, chunk_size, threads, delay, use_cpulimit, cpulimit)

    if use_cpulimit:
        # Run the script with CPU limit using cpulimit tool
        calculator.run_with_cpulimit()
    else:
        # Load data
        df_selection = calculator.load_data()

        # Filter out already processed SMILES strings
        df_selection = calculator.filter_processed_smiles(df_selection)

        # Calculate and save descriptors chunk by chunk
        calculator.calculate_descriptors(df_selection['canonical_smiles'].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PaDEL descriptors with CPU usage control options.")
    parser.add_argument('input_file', type=str, help="Input CSV file containing canonical_smiles.")
    parser.add_argument('output_file', type=str, help="Output CSV file to save the PaDEL descriptors.")
    parser.add_argument('--chunk_size', type=int, default=256, help="Number of SMILES to process in each chunk.")
    parser.add_argument('--threads', type=int, default=10, help="Number of threads to use for PaDEL descriptor calculation.")
    parser.add_argument('--delay', type=int, default=1, help="Seconds to wait between chunk processing to throttle CPU usage.")
    parser.add_argument('--use_cpulimit', action='store_true', help="If set, runs the script with cpulimit to restrict CPU usage.")
    parser.add_argument('--cpulimit', type=int, default=90, help="CPU limit percentage to use with cpulimit (if --use_cpulimit is set).")
    
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.chunk_size, args.threads, args.delay, args.use_cpulimit, args.cpulimit)

