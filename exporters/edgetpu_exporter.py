import subprocess


def export_edgetpu(tflite_model, out_dir):
    bashCmd = "edgetpu_compiler --out_dir {} {}".format(out_dir, tflite_model)
    process = subprocess.Popen(bashCmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    output, error = process.communicate()
    if process.returncode == 0:
        model_name = ".".join(tflite_model.split("/")[-1].split(".")[:-1])
        print("your model compiled sucessfully under: {}".format(out_dir + model_name + "_edgetpu.tflite"))
    else:
        print("an error has occurred")

