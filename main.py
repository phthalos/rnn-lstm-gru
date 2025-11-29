import torch
from torch.optim import optimizer
from torch.utils.data.dataset import random_split
import re
from collections import Counter, OrderedDict
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np

# 1. Load dataset and build vocab
# (X_train, y_train), (X_test, y_test) = imdb.load_data("imdb")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)
vocab_size = 20000

import torch.nn as nn
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)

train_texts = X_train
train_labels = y_train
test_texts = X_test
test_labels = y_test

vocab = imdb.get_word_index()
index_to_word = {}
for key, value in vocab.items():
  index_to_word[value+3] = key
# print(index_to_word[4])
# 빈도수 상위 1등 단어 : the
print('빈도수 상위 1등 단어 : {}'.format(index_to_word[4]))

# 2. Encode texts to fixed-length sequences
def encode(text, vocab, max_len=100):
    tokens = text.lower().split()
    idxs = [vocab.get(token, 0) for token in tokens]  # 0 for unknown words
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs

# 3. Custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# 데이터셋 클래스
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# train/valid 나누기
np.random.seed(1)
indices = np.arange(len(train_texts))
np.random.shuffle(indices)
train_idx, valid_idx = indices[:20000], indices[20000:25000]

train_dataset = IMDBDataset([train_texts[i] for i in train_idx],
                            [train_labels[i] for i in train_idx])
valid_dataset = IMDBDataset([train_texts[i] for i in valid_idx],
                            [train_labels[i] for i in valid_idx])
test_dataset = IMDBDataset(test_texts, test_labels)


# for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
#   index_to_word[index] = token
#
# # <sos> this film was just brilliant casting location ...(생략)
# print(' '.join([index_to_word[index] for index in X_train[0]]))

# 영화 리뷰 데이터 준비
# 각 세트는 25,000개의 샘플을 가짐
# 긍정적 리뷰(1) 부정적 리뷰(0)
# train_dataset = IMDB(split='train')
# test_dataset = IMDB(split='test')
#
# ## 데이터셋 만들기
# # 훈련: 전체 개수의 80% 검증 20%로 샘플을 나눔. 샘플은 랜덤하게 선택함.
# torch.manual_seed(1)
# train_dataset, valid_dataset = random_split(
#     list(train_dataset),[20000,5000])
# #
# # # 입력 데이터 준비
# # ## 고유 토큰(단어) 찾기
# # # HTML 마크업, 구둣점, 글자가 아닌 다른 문자 제거
# def tokenizer(text):
#     text = re.sub('<[^>]*>','',text)
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
#     text = re.sub('[\W]+',' ',text.lower()) + \
#            ' '.join(emoticons).replace('-','')
#     tokenized = text.split()
#     return tokenized
# #
# # # 토큰 개수 출력
# token_counts = Counter()
# for label, line in train_dataset:
#     tokens = tokenizer(line)
#     token_counts.update(tokens)
# print('어휘 사전 크기: ',len(token_counts))
#
# ## 고유 토큰을 정수로 인코딩
# sorted_by_freq_tuples = sorted(
#     token_counts.items(), key=lambda x: x[1], reverse=True
# )
# ordered_dict = OrderedDict(sorted_by_freq_tuples)
# vocab = vocab(ordered_dict)
# vocab.insert_token("<pad>",0)
# vocab.insert_token("<unk>",1)
# vocab.set_default_index(1)
#
# # 샘플 입력을 정수로 변환하는 예제
# # print([vocab[token] for token in ['this','is','an','example']])
#
## 데이터셋의 텍스트를 변환하는 함수
# pos와 neg를 1 또는 0으로 바꿈
# text_pipeline =\
#     lambda x: [vocab[token] for token in tokenizer(x)]
# label_pipeline = lambda x: 1. if x == 'pos' else 0.
#
#
## 텍스트 인코딩, 레이블 변환함수
# def collate_batch(batch):
#     label_list, text_list, lengths = [],[],[]
#     for _label, _text in batch:
#         label_list.append(label_pipeline(_label))
#         procecssed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
#         text_list.append(procecssed_text)
#         lengths.append(procecssed_text.size(0))
#     label_list = torch.tensor(label_list)
#     lengths = torch.tensor(lengths)
#     padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
#     return padded_text_list, label_list, lengths
#
# ## 샘플의 배치 생성
from torch.utils.data import DataLoader
# dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
#
# text_batch, label_batch, length_batch = next(iter(dataloader))
# # 개별 원소의 크기 출력
# # 미니 배치에 있는 시퀀스는 효율적으로 텐서에 저장하기 위해 동일한 길이가 되어야 하지만, 지금은 시퀀스 길이가 다르다.
# # print(text_batch)
# # print(label_batch)
# # print(length_batch)
# # print(text_batch.shape)
#
# # 4개 중 가장 큰 크기의 샘플을 제외한 나머지 3개의 데이터셋을 배치 크기가 32인 데이터 로더로 만듦
# # RNN 모델에 적합한 포맷으로 변환 완료
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    # 길이가 다른 시퀀스 패딩 (0으로)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    return texts_padded, labels, lengths

batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
#
# # 임베딩 (단어 벡터의 차원 줄이기)
embedding = nn.Embedding(num_embeddings=10,embedding_dim=3,padding_idx=0)
# 4개의 인덱스를 가진 샘플 2개로 구성된 배치
text_encoded_input = torch.LongTensor([[1,2,4,5],[4,3,2,0]])
# print(embedding(text_encoded_input))

## 전처리 완료

# ## RNN 모델 만들기
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, hidden = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
print(model)
#
# # GPU 사용 가능 여부 확인 (CUDA: 엔비디아, MPS: 맥북 M1/M2, CPU: 그 외)
# # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# # print(f"현재 사용 중인 디바이스: {device}")
#
# # 모델을 생성한 후 .to(device)를 붙여서 GPU 메모리에 올림
# # model = model.to(device)
#
# 한 epoch동안 주어진 데이터셋에서 모델을 훈련하고, 분류 정확도와 손실을 반환하는 함수
def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0

    # tqdm으로 dataloader를 감싸서 진행바 생성
    # 제목: Training
    pbar = tqdm(dataloader, desc='Training')

    for text_batch, label_batch, lengths in dataloader:
        # GPU로 데이터 이동
        # text_batch = text_batch.to(device)
        # label_batch = label_batch.to(device)

        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)

        loss.backward()
        optimizer.step()

        # 정확도 및 손실 계산
        batch_acc = ((pred >= 0.5).float() == label_batch).float().sum().item()

        total_acc += (
        (pred >= 0.5).float() == label_batch
        ).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)

        # 진행바 오른쪽 끝에 실시간 Loss와 정확도 표시
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': batch_acc / len(label_batch)
        })

    return total_acc / len(dataloader.dataset), total_loss/len(dataloader.dataset)

# 모델 성능 평가
def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            # GPU로 데이터 이동
            # text_batch = text_batch.to(device)
            # label_batch = label_batch.to(device)

            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += (
            (pred >= 0.5).float() == label_batch
            ).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)
        return total_acc / len(list(dataloader.dataset)), total_loss / len(list(dataloader.dataset))

# 이진 분류의 경우 이진 크로스 엔트로피 손실함수 사용
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 10회의 epoch 동안 모델 훈련
num_epochs = 10
torch.manual_seed(1)
# for epoch in range(num_epochs):
#     acc_train, loss_train = train(train_dl)
#     # 검증 성능 표시
#     acc_valid, loss_valid = evaluate(valid_dl)
#     print(f'에포크 {epoch} 정확도: {acc_train: .4f}'
#           f' 검증 정확도: {acc_valid: .4f}')
#
# acc_test, _ = evaluate(test_dl)
# print(f'테스트 정확도: {acc_test: .4f}')

# 결과 저장을 위한 리스트 초기화
history = {
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': []
}

# 10회의 epoch 동안 모델 훈련
print("학습 시작...")
for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)

    # 매 epoch 결과를 리스트에 저장
    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['valid_loss'].append(loss_valid)
    history['valid_acc'].append(acc_valid)

    print(f'에포크 {epoch + 1}/{num_epochs} | '
          f'Train Loss: {loss_train:.4f}, Acc: {acc_train:.4f} | '
          f'Valid Loss: {loss_valid:.4f}, Acc: {acc_valid:.4f}')

# 테스트 데이터 평가
acc_test, _ = evaluate(test_dl)
print(f'\n최종 테스트 정확도: {acc_test:.4f}')

# 그래프 출력
epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss')
plt.plot(epochs_range, history['valid_loss'], 'r--', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_acc'], 'b-', label='Training Accuracy')
plt.plot(epochs_range, history['valid_acc'], 'r--', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
