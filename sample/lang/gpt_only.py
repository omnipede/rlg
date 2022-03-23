import os
from pathlib import Path
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import matplotlib.pyplot as plt


if __name__ == '__main__':

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    model: GPTNeoForCausalLM = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    l_path = '/Users/omnipede/data/t.csv'
    losses = []
    act = []

    with open(l_path) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx > 100:
                break

            print(idx)

            splitted = line.split(',')
            one = splitted[0]
            two = splitted[1]
            inputs_list = tokenizer([one, two], return_tensors="pt", padding=True)
            input_ids = inputs_list.input_ids[0]
            input_ids2 = inputs_list.input_ids[1]

            outputs = model(input_ids=input_ids, labels=input_ids2)
            losses.append(outputs.loss.item())

            act_output = model(input_ids=input_ids, labels=input_ids)
            act.append(act_output.loss.item())

    plt.plot(losses, color='red')
    plt.plot(act, color='blue')
    plt.show()

    #
    # # Read all lyrics from disk
    # clip_map = {}
    #
    # data_path = "/Users/omnipede/data"
    # dirs = os.listdir(data_path)
    # for song in dirs:
    #     song_dir = os.path.join(data_path, song)
    #     if os.path.isdir(song_dir) is False:
    #         continue
    #
    #     clip_map[song] = []
    #
    #     subclips_path = os.path.join(song_dir, 'clips')
    #     subclips = [str(x) for x in sorted(Path(subclips_path).iterdir(), key=os.path.getmtime)]
    #
    #     for subclip in subclips:
    #         # Txt file 만 가져온다.
    #         if subclip.endswith(".txt"):
    #             clip_map[song].append(subclip)
    #
    # lyrics = []
    #
    # for song, subclips in clip_map.items():
    #     for i in range(len(subclips)):
    #
    #         lyrics.append({
    #             # Save target, next
    #             't': subclips[i],
    #             'n': subclips[(i + 1) % len(subclips)]
    #         })
    #
    # lyric_file_path = os.path.join(data_path, 't.csv')
    # print(lyric_file_path)
    # with open(lyric_file_path, 'w') as lyric_file:
    #
    #     for lyric in lyrics:
    #         target = lyric['t']
    #         nxt = lyric['n']
    #
    #         with open(target, 'r') as f:
    #             target_lyric = f.read()
    #
    #         with open(nxt, 'r') as f:
    #             nxt_lyric = f.read()
    #
    #         lyric_file.write(target_lyric + ',' + nxt_lyric + '\n')
