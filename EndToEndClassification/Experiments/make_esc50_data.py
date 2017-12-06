from EndToEndClassification.Dataset.esc50_processor import ESC50Processor
import argparse

description = 'process the ESC50 dataset'

parser = argparse.ArgumentParser(description=description)
parser.add_argument("esc_50_path", type=str)
parser.add_argument("destination_folder", type=str)

args = parser.parse_args()

ESC50Processor(args.esc_50_path, args.destination_folder)
