import numpy as np

wv = np.load('../data/output/model_word2vec_200.npy', allow_pickle=True).item()

embeddings = []
inp_text = "../data/ijcnlp_dailydialog/dialogues_text.txt"
inp_label = "../data/ijcnlp_dailydialog/dialogues_emotion.txt"
outp = "../data/output/sample"
f_text = open(inp_text, encoding='utf-8')
f_label = open(inp_label, encoding='utf-8')
text_num = []
label_num = []

for dialogue in f_text:
    lines = dialogue.split('__eou__')
    text_num.append(len(lines))
    for line in lines:
        if line == '\n':
            continue
        words = line.strip().split(' ')
        text_embedding = np.zeros(200)
        for word in words:
            try:
                text_embedding += wv[word]
            except KeyError:
                text_embedding += wv['UNK']
        embeddings.append(text_embedding)

embeddings = np.asarray(embeddings, dtype=np.float32)

labels = []
for dialogue in f_label:
    lines = dialogue.split(' ')
    label_num.append(len(lines))
    for line in lines:
        if line == '\n':
            continue
        labels.append(line)


labels = np.asarray(labels, dtype=np.int32)

np.savez(outp, x=embeddings, y=labels)


f_text.close()
f_label.close()
