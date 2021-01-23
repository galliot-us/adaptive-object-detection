import subprocess
import os
import argparse


def export_edgetpu(tflite_model, out_dir):
    """
    Compile quantized tflite models to the Edge TPU. note that this module can not run on ARM devices like Coral Dev Board.
    Args:
        tflite_model: Path of the input tflite file
        out_dir: Directory to store the output compiled file
    """
    if not os.path.isfile(tflite_model):
        raise FileNotFoundError("the provided tflite file : {0} is not exist".format(tflite_model))
    if not os.path.isdir(out_dir):
        print("the provided output directory : {0} is not exist".format(out_dir))
        print("creating output directory : {0}".format(out_dir))
        os.makedirs(out_dir, exist_ok=True)

    bashCmd = "edgetpu_compiler --out_dir {0} {1}".format(out_dir, tflite_model)
    process = subprocess.Popen(bashCmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    output, error = process.communicate()
    if process.returncode == 0:
        model_name = ".".join(tflite_model.split("/")[-1].split(".")[:-1])
        print("your model compiled sucessfully under: {0}".format(os.path.join(out_dir, model_name + "_edgetpu.tflite")))
    else:
        print("an error has occurred")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script compile a quantized tflite model to an EdgeTPU model")
    parser.add_argument("--tflite_file", type=str, required=True, help="the path of input tflite file")
    parser.add_argument("--out_dir", type=str, required=True, help="a directory to store the output compiled file")
    args = parser.parse_args()
    tflite_model = args.tflite_file
    out_dir = args.out_dir
    export_edgetpu(tflite_model=tflite_model, out_dir=out_dir)

