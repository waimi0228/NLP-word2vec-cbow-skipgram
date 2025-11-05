import jieba
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# 中文語料
content = """竹子湖海芋季揭幕 白色情人節浪漫賞花 
2025 年竹子湖海芋季今（14）日揭開序幕，白色海芋花田進入盛開期，北市府
與北投區農會攜手輔導竹子湖農園，今年以「芋想世界」為花季主題風格，精
彩活動包含例假日免費生態人文導覽解說、免費印卡讚及千元有找的竹子湖生
態農園體驗等，海芋季舉行至4月27日止。 
今天是白色情人節，海芋象徵真誠、純潔、純白的愛情花語，花季開幕活動
上，來賓在象徵傳遞心意的明信片留下愛的宣言，祝福每一對情侶，愛如花般
盛開「芋見幸福」。 
今年花季以「芋想世界」為主題風格，花農們親手打造別出心裁的海芋地景花
藝設計。4/12-13、5/3、5/31、6/1有「巴島柑嬤店」市集，每天下午有海芋 · 
繡球印卡讚活動，可現場印出自己拍攝的拍立得照片。3月15日至4月17日
假日上午10點半和下午2點，各有1場生態人文導覽解說，無須報名，直接
前往竹子湖入口停車場報到。 
台北市副市長張溫德強調，全台唯一最大的純白海芋田就在北投竹子湖，從3
月開始一直到6月，接續有海芋季及繡球花季，是台北市民春夏季踏青休憩的
首選，也吸引全台各地的愛花人士來訪。 
竹子湖海芋季從即日起至4月27日舉行，繡球花季則接棒於5月23日至6月
22 日舉行，花季期間產業局規劃系列精彩活動，包含例假日免費生態人文導覽
解說及千元有找竹子湖生態農園體驗等。遊客能深度體驗竹子湖「賞花、溪
遊、走讀、野宴」之旅，包括採海芋、導覽解說、品嚐山產野菜、下午茶、花
藝DIY及購買園藝花卉等，美麗饗宴一次滿足。 
有關「2025竹子湖海芋季」花況及交通資訊，可至海芋季官網
（www.callalily.com.tw）查詢。活動期間例假日仰德大道及竹子湖地區實施管
制，民眾自台北車站可搭乘260路公車，於「陽明山公車總站」下車，再轉乘
108、124，並於「竹子湖派出所」公車站下車轉乘或步行前往竹子湖。"""

# 分行並使用 jieba 斷詞
sentences = [line.strip() for line in corpus.splitlines() if line.strip()]
tokenized_sentences = [list(jieba.cut(sentence)) for sentence in sentences]
print("斷詞結果：")
for tokens in tokenized_sentences:
    print(tokens)

# 建立字典
all_tokens = [w for tokens in tokenized_sentences for w in tokens]
vocab = sorted(set(all_tokens))
# 從 1 開始編號，保留 0 為 padding 用
word_to_index = {w: i+1 for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}
vocab_size = len(vocab) + 1  # 包含 0 的索引

# 4. 準備 CBOW 的訓練資料 (以滑動窗口處理各句子)
context_window = 2  # 上下文窗口大小（左右各2個詞）
X, y = [], []

for tokens in tokenized_sentences:
    token_indices = [word_to_index[w] for w in tokens]
    for i in range(len(token_indices)):
        context = []
        # 前 context_window 個詞
        for j in range(max(0, i - context_window), i):
            context.append(token_indices[j])
        # 後 context_window 個詞
        for j in range(i + 1, min(len(token_indices), i + context_window + 1)):
            context.append(token_indices[j])
        # 如果上下文不足，則以 0 做 padding
        if len(context) < 2 * context_window:
            context = context + [0] * (2 * context_window - len(context))
        X.append(context)
        y.append(token_indices[i])

X = np.array(X)
y = np.array(y)
y_categorical = to_categorical(y, num_classes=vocab_size)

# 為了用產生器方式訓練，將所有 token 轉為一個連續的序列
corpus_indices = [word_to_index[w] for w in all_tokens]

# 定義一個產生器函式，依序回傳 (x, y) 訓練資料
def generate_context_word_pairs(corpus, window_size, vocab_size):
    for i in range(window_size, len(corpus) - window_size):
        center_word = corpus[i]
        context_indices = []
        for j in range(i - window_size, i):
            context_indices.append(corpus[j])
        for j in range(i + 1, i + window_size + 1):
            context_indices.append(corpus[j])
        x = np.array(context_indices).reshape(1, -1)
        y = np.zeros((1, vocab_size), dtype=np.float32)
        y[0, center_word] = 1.0
        yield x, y

# 建立 CBOW 模型 (使用 Sequential)
embed_size = 100  # 詞向量維度
window_size = context_window  # 統一變數名稱

cbow = Sequential()
# Embedding 層：輸入大小為 vocab_size，輸出向量維度為 embed_size
# input_length 為左右各 window_size，即 2 * window_size
cbow.add(Embedding(input_dim=vocab_size,
                   output_dim=embed_size,
                   input_length=window_size * 2,
                   name='embedding_layer'))
# Lambda 層：取上下文各詞向量的平均
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
# Dense 層：輸出大小為 vocab_size，並以 softmax 預測中心詞
cbow.add(Dense(vocab_size, activation='softmax', name='output_layer'))

cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print(cbow.summary())

# 訓練模型
# 方法1：使用 fit() 進行訓練
print("使用 fit() 訓練模型...")
cbow.fit(X, y_categorical, epochs=100, verbose=1)


# 取出 Embedding 層權重作為詞向量
embeddings = cbow.get_layer('embedding_layer').get_weights()[0]  # shape: (vocab_size, embed_size)

from sklearn.metrics.pairwise import euclidean_distances

# 計算歐幾里得距離矩陣
distance_matrix = euclidean_distances(embeddings)
print(distance_matrix.shape)  # 例如 (10, 10)

# 選擇一些想查詢的詞，必須確定它們都在字典中
search_terms = ["竹子湖", "海芋", "北投", "愛情", "幸福"]

# 建立一個字典 similar_words
similar_words = {
    term: [
        index_to_word[idx]
        for idx in distance_matrix[word_to_index[term]].argsort()[1:6]
    ]
    for term in search_terms
}

print(similar_words)
