import os
from glob import glob

inp_dir = "/home/fm001/zhangyuqiang_check/workspace/fork/StyleTTS/LJSpeech-1.1/wavs"
out_dir = "/home/fm001/zhangyuqiang_check/workspace/fork/StyleTTS/LJSpeech-1.1-tmp/wavs"

def func(wav_path):
    filename = wav_path.split("/")[-1]
    out_path = os.path.join(out_dir, filename)
    cmd = "ffmpeg -i {} -ar 24000 {}".format(wav_path, out_path)
    os.system(cmd)

wav_paths = glob(inp_dir + "/*wav")
print(len(wav_paths))
for wav_path in wav_paths:
    func(wav_path)

