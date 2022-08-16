from dadagp import dadagp_decode
import sys
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_folder.py INPUT_FOLDER OUTPUT_FOLDER")
        return

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if not os.path.exists(input_folder):
        print("Input folder {} does not exist".format(input_folder))
        return
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    for item in os.listdir(input_folder):
        print("Decoding {}...".format(item))
        full_path = os.path.join(input_folder, item)
        if full_path.endswith(".txt"):
            output_path = os.path.join(output_folder, item[:-4] + ".gp5")
            dadagp_decode(full_path, output_path)

if __name__ == "__main__":
    main()