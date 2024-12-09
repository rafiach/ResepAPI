from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
import pickle
import pandas as pd

# Load model dan vectorizer dari pickle
with open('model/knn_model.pkl', 'rb') as model_file:
    knn = pickle.load(model_file)

with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load data resep
data = pd.read_csv('model/resepfinal834.csv')

@api_view(['GET', 'POST'])
def rekomendasi_resep(request):
    input_bahan = request.query_params.get('bahan', '') if request.method == 'GET' else request.data.get('bahan', '')

    if not input_bahan:
        return Response({"error": "Input bahan tidak ditemukan"}, status=400)

    try:
        # Proses rekomendasi
        input_bahan_vec = tfidf_vectorizer.transform([input_bahan])
        distances, indices = knn.kneighbors(input_bahan_vec, n_neighbors=5)

        hasil_rekomendasi = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            hasil_rekomendasi.append({
                'Nama_Resep': data.iloc[idx]['Title'],
                'Bahan': data.iloc[idx]['Ingredients Cleaned'],
                'Langkah_Pembuatan': data.iloc[idx]['Steps'],
                'Kesamaan': 1 - distances[0][i]
            })

        # Return rekomendasi dalam format JSON
        return Response(hasil_rekomendasi)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
