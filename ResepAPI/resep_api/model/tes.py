import pickle

# Memuat vectorizer dari file pickle
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Test input
test_input = "tomat, bawang"

# Transformasi input menggunakan tfidf_vectorizer
test_vec = tfidf_vectorizer.transform([test_input])

# Menampilkan bentuk hasil transformasi
print(test_vec.shape)  # Harus menghasilkan bentuk yang valid
