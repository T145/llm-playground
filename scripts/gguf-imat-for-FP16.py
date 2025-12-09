import os
import shutil
import subprocess
import zipfile

import requests
from huggingface_hub import snapshot_download


def clone_or_update_llama_cpp():
    print("Preparing...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    if not os.path.exists("llama.cpp"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp"])
    else:
        os.chdir("llama.cpp")
        subprocess.run(["git", "pull"])
    os.chdir(base_dir)
    print("The 'llama.cpp' repository is ready.")

def download_llama_release():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dl_dir = os.path.join(base_dir, "bin", "dl")
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)

    os.chdir(dl_dir)
    latest_release_url = "https://github.com/ggerganov/llama.cpp/releases/latest"
    response = requests.get(latest_release_url)
    if response.status_code == 200:
        latest_release_tag = response.url.split("/")[-1]
        download_url = f"https://github.com/ggerganov/llama.cpp/releases/download/{latest_release_tag}/llama-{latest_release_tag}-bin-win-cuda-cu12.2.0-x64.zip"
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(f"llama-{latest_release_tag}-bin-win-cuda-cu12.2.0-x64.zip", "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(f"llama-{latest_release_tag}-bin-win-cuda-cu12.2.0-x64.zip", "r") as zip_ref:
                zip_ref.extractall(os.path.join(base_dir, "bin"))
            print("Downloading latest 'llama.cpp' prebuilt Windows binaries...")
            print("Download and extraction completed successfully.")
            return latest_release_tag
        else:
            print("Failed to download the release file.")
    else:
        print("Failed to fetch the latest release information.")

def download_cudart_if_necessary(latest_release_tag):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cudart_dl_dir = os.path.join(base_dir, "bin", "dl")
    if not os.path.exists(cudart_dl_dir):
        os.makedirs(cudart_dl_dir)

    cudart_zip_file = os.path.join(cudart_dl_dir, "cudart-llama-bin-win-cu12.2.0-x64.zip")
    cudart_extracted_files = ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]

    if all(os.path.exists(os.path.join(base_dir, "bin", file)) for file in cudart_extracted_files):
        print("Cuda resources already exist. Skipping download.")
    else:
        cudart_download_url = f"https://github.com/ggerganov/llama.cpp/releases/download/{latest_release_tag}/cudart-llama-bin-win-cu12.2.0-x64.zip"
        response = requests.get(cudart_download_url)
        if response.status_code == 200:
            with open(cudart_zip_file, "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(cudart_zip_file, "r") as zip_ref:
                zip_ref.extractall(os.path.join(base_dir, "bin"))
            print("Preparing 'cuda' resources...")
            print("Download and extraction of cudart completed successfully.")
        else:
            print("Failed to download the cudart release file.")

def download_model_repo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_id = input("Enter the model ID to download (e.g., huggingface/transformers): ")
    model_name = model_id.split("/")[-1]
    model_dir = os.path.join(models_dir, model_name)

    gguf_dir = os.path.join(base_dir, "models", f"{model_name}-GGUF")
    gguf_model_path = os.path.join(gguf_dir, f"{model_name}-F16.gguf")
    imatrix_file_name = input("Enter the name of the imatrix.txt file (default: imatrix.txt): ").strip() or "imatrix.txt"
    delete_model_dir = input("Remove HF model folder after converting original model to GGUF? (yes/no) (default: no): ").strip().lower()

    if os.path.exists(gguf_model_path):
        create_imatrix(base_dir, gguf_dir, gguf_model_path, model_name, imatrix_file_name)
    else:
        if os.path.exists(model_dir):
            print("Model repository already exists. Using existing repository.")

            convert_model_to_gguf_f16(base_dir, model_dir, model_name, delete_model_dir, imatrix_file_name)

        else:
            revision = input("Enter the revision (branch, tag, or commit) to download (default: main): ") or "main"

            print("Downloading model repository...")
            snapshot_download(repo_id=model_id, local_dir=model_dir, revision=revision)
            print("Model repository downloaded successfully.")

            convert_model_to_gguf_f16(base_dir, model_dir, model_name, delete_model_dir, imatrix_file_name)

def convert_model_to_gguf_f16(base_dir, model_dir, model_name, delete_model_dir, imatrix_file_name):
    convert_script = os.path.join(base_dir, "llama.cpp", "convert_hf_to_gguf.py")
    gguf_dir = os.path.join(base_dir, "models", f"{model_name}-GGUF")
    gguf_model_path = os.path.join(gguf_dir, f"{model_name}-F16.gguf")

    if not os.path.exists(gguf_dir):
        os.makedirs(gguf_dir)

    if not os.path.exists(gguf_model_path):
        subprocess.run(["python", convert_script, model_dir, "--outfile", gguf_model_path, "--outtype", "f16"])

        if delete_model_dir == "yes" or delete_model_dir == "y":
            shutil.rmtree(model_dir)
            print(f"Original model directory '{model_dir}' deleted.")
        else:
            print(f"Original model directory '{model_dir}' was not deleted. You can remove it manually.")


    create_imatrix(base_dir, gguf_dir, gguf_model_path, model_name, imatrix_file_name)

def create_imatrix(base_dir, gguf_dir, gguf_model_path, model_name, imatrix_file_name):
    imatrix_exe = os.path.join(base_dir, "bin", "llama-imatrix.exe")
    imatrix_output_src = os.path.join(gguf_dir, "imatrix.dat")
    imatrix_output_dst = os.path.join(gguf_dir, "imatrix.dat")
    if not os.path.exists(imatrix_output_dst):
        try:
            subprocess.run([imatrix_exe, "-m", gguf_model_path, "-f", os.path.join(base_dir, "imatrix", imatrix_file_name), "-ngl", "8"], cwd=gguf_dir)
            shutil.move(imatrix_output_src, imatrix_output_dst)
            print("imatrix.dat moved successfully.")
        except Exception as e:
            print("Error occurred while moving imatrix.dat:", e)
    else:
        print("imatrix.dat already exists in the GGUF folder.")

    quantize_models(base_dir, model_name)

def quantize_models(base_dir, model_name):
    gguf_dir = os.path.join(base_dir, "models", f"{model_name}-GGUF")
    f16_gguf_path = os.path.join(gguf_dir, f"{model_name}-F16.gguf")

    quantization_options = [
        "IQ3_M", "IQ3_XXS",
        "Q4_K_M", "Q4_K_S", "IQ4_XS",
        "Q5_K_M", "Q5_K_S",
        "Q6_K",
        "Q8_0"
    ]

    for quant_option in quantization_options:
        quantized_gguf_name = f"{model_name}-{quant_option}-imat.gguf"
        quantized_gguf_path = os.path.join(gguf_dir, quantized_gguf_name)
        quantize_command = os.path.join(base_dir, "bin", "llama-quantize.exe")
        imatrix_path = os.path.join(gguf_dir, "imatrix.dat")

        subprocess.run([quantize_command, "--imatrix", imatrix_path,
                        f16_gguf_path, quantized_gguf_path, quant_option], cwd=gguf_dir)
        print(f"Model quantized with {quant_option} option.")

def main():
    clone_or_update_llama_cpp()
    latest_release_tag = download_llama_release()
    download_cudart_if_necessary(latest_release_tag)
    download_model_repo()
    print("Finished preparing resources.")

if __name__ == "__main__":
    main()
