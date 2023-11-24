from glob import glob
wav_path_list = glob("ref_wav/LibriTTS-train-clean-360-sub/*/*wav")
print(len(wav_path_list))
with open("./Data/val_list_libritts_seen.txt", "w") as log:
    for path in wav_path_list:
        line = path + "|something|something\n"
        log.write(line)