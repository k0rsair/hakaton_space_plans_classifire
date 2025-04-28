from flask import Flask, request, jsonify
import numpy as np
import joblib


app = Flask(__name__)

cluster_to_topic = {
    0: "спорт",
    1: "юмор",
    2: "реклама",
    3: "соцсети",
    4: "политика",
    5: "личная_жизнь"
}

# Загрузка модели (например, TensorFlow)
tfidf = joblib.load('tfidf_vectorizer.joblib')
model_tfidf = joblib.load('model_tfidf.joblib')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из запроса
        data = request.get_json()
        # Преобразование данных в формат, который модель может обработать
        input_data = list([data['text']])  # пример преобразования данных
        NEW_X = tfidf.transform(input_data)

        # Предсказываем
        predictions = model_tfidf.predict(NEW_X)
        # Отправка предсказания как JSON-ответ

        return_array = []
        key_item = 0
        for value in predictions.tolist()[0]:
            if value:
                return_array.append(cluster_to_topic[key_item])
            key_item += 1
        return return_array
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
