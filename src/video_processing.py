import subprocess

input_file = "../temp_dataset/videoplayback.mp4"

for i in range(4):
    start = i * 15 * 60
    output = f"../temp_dataset/output_part_{i+1}.mp4"

    subprocess.run([
        "ffmpeg",
        "-ss", str(start),
        "-i", input_file,
        "-t", str(15*60),
        "-c", "copy",
        output
    ])