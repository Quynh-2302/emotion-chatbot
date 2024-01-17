import numpy as np

# Load trọng số embedding từ file đã được lưu trước đó
wv = np.load('../data/output/model_word2vec_200.npy', allow_pickle=True).item()

# Khởi tạo danh sách để chứa các vector embedding và nhãn tương ứng
embeddings = []
inp_text = "../data/ijcnlp_dailydialog/dialogues_text.txt"
inp_label = "../data/ijcnlp_dailydialog/dialogues_emotion.txt"
outp = "../data/output/sample_seq"
f_text = open(inp_text, encoding='utf-8')
f_label = open(inp_label, encoding='utf-8')
text_num = []
label_num = []

# Đọc và xử lý dữ liệu văn bản
for dialogue in f_text:
    lines = dialogue.split('__eou__')
    text_num.append(len(lines))
    for line in lines:
        if line == '\n':
            continue
        words = line.strip().split(' ')
        text_sequence = []

        # Chuyển từng từ trong câu thành vector embedding
        for word in words:
            try:
                text_sequence.append(wv[word])
            except KeyError:
                # Sử dụng vector cho từ không có trong từ điển
                text_sequence.append(wv['UNK'])

        embeddings.append(text_sequence)

# Đọc và xử lý nhãn cảm xúc
labels = []
for dialogue in f_label:
    lines = dialogue.split(' ')
    label_num.append(len(lines))
    for line in lines:
        if line == '\n':
            continue
        labels.append(line)

# Chuyển đổi nhãn thành numpy array có kiểu dữ liệu int32
labels = np.asarray(labels, dtype=np.int32)

# Tìm độ dài tối thiểu của các vector embedding
min_length = min(len(seq) for seq in embeddings)

# Cắt các vector embedding để có cùng độ dài
embeddings = [seq[:min_length] for seq in embeddings]

# Chuyển đổi danh sách thành mảng numpy
embeddings = np.asarray(embeddings, dtype=np.float32)

# Lưu dữ liệu đã xử lý thành file npz
np.savez(outp, x=embeddings, y=labels)
print("Data saved successfully to:", outp + ".npz")

# Đóng tệp văn bản
f_text.close()
f_label.close()
