import argparse

parser = argparse.ArgumentParser()

# Parse arguments
parser.add_argument('--feature_model_type', type=str, help='nr of batches')
parser.add_argument('-f', type=str, help='Path to kernel json')

# Extract args
ARGS, unknown = parser.parse_known_args()
