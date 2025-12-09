import os
import subprocess
import sys

sys.dont_write_bytecode = True


# https://www.reddit.com/r/ollama/comments/1f9mx9n/why_does_ollama_default_to_q_4_when_q4_k_m_and_q4/
def main():
    model_id = input("Model ID: ")
    wd = os.getcwd()
    out_path = os.path.join(wd, "output")
    f16 = f"/models/{model_id}.f16.gguf"

    with subprocess.Popen(f"docker run --rm --gpus all -v {out_path}:/models/ -e CUDA_VERSION=12.7.0 ghcr.io/ggerganov/llama.cpp:full-cuda --convert --outtype f16 --outfile {f16} /models/{model_id}", shell=True, stdout=subprocess.PIPE).stdout as out:
        print(out.read())

    quants = ["Q8_0", "Q4_K_M"]

    for quant in quants:
        target = f"{out_path}/{model_id}.{quant}.gguf"

        if not os.path.isfile(target):
            with subprocess.Popen(f"docker run --rm --gpus all -v {out_path}:/models/ -e CUDA_VERSION=12.7.0 ghcr.io/ggerganov/llama.cpp:full-cuda --quantize {f16} /models/{model_id}.{quant}.gguf {quant}", shell=True, stdout=subprocess.PIPE).stdout as out:
                print(out.read())

    #os.remove(f16)


if __name__ == "__main__":
    main()
