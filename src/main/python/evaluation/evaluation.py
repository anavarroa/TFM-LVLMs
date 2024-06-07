import subprocess

def run_llava_serve():
    command = [
        "python",
        "-m",
        "llava.serve.cli",
        "--model-path",
        "liuhaotian/llava-v1.5-7b",
        "--image-file",
        "data/imagenes/NWPU/NWPU_images/airplane/airplane_009.jpg",
        "--load-4bit"
    ]
    subprocess.run(command)

if __name__ == "__main__":
    run_llava_serve()
