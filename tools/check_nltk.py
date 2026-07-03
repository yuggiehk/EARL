import os
import nltk
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# ✅ 显示当前 NLTK 搜索路径
print("NLTK data search path:")
print(nltk.data.path)

# ✅ 检查 punkt
try:
    nltk.data.find('tokenizers/punkt')
    print(" punkt ✅ found")
except LookupError:
    print(" punkt ❌ NOT found")

# ✅ 检查 wordnet
try:
    nltk.data.find('corpora/wordnet')
    print(" wordnet ✅ found")
except LookupError:
    print(" wordnet ❌ NOT found")

# ✅ 测试 WordNet
try:
    syns = wordnet.synsets("car")
    print(f" wordnet loaded, 'car' synsets count: {len(syns)}")
except Exception as e:
    print(" wordnet ❌ load error:", e)

# ✅ 测试分词和 METEOR
try:
    ref = "the car is driving fast"
    cand = "the car drives quickly"
    ref_tokens = word_tokenize(ref)
    cand_tokens = word_tokenize(cand)
    print(meteor_score([ref_tokens], cand_tokens))
    score = meteor_score([ref_tokens], cand_tokens)
    print(f" METEOR test ✅ score: {score:.4f}")
except Exception as e:
    print(" METEOR ❌ test error:", e)