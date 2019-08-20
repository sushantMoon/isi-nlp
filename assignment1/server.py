from flask import Flask, render_template, g, request, jsonify
from query_prediction_lstm import lstm
import os
import tensorflow as tf
from n_gram import ngram_language_model


app = Flask(
    __name__,
    static_url_path='',
    static_folder='templates',
    template_folder='templates'
    )


@app.route('/predict-next-words')
def make_predictions():
    text = request.args.get('message').strip()
    cleaned = ' '.join([x.strip() for x in text.split()])
    print("Input : \n{}".format(cleaned))
    with graph.as_default():
        predictions = lstm_model.get_prediction(message=cleaned, quantity=5)
    predictions.update(
        ngram_model.make_predictions(
            cleaned,
            number_of_predictions=5
        )
    )
    print("Predictions : {}".format(predictions))
    return jsonify(predictions)


@app.route('/')
def index_page():
    return render_template("index.html")

if __name__ == '__main__':
    global lstm_model, graph, ngram_model
    lstm_model = lstm()
    graph = tf.get_default_graph()
    ngram_model = ngram_language_model()
    ngram_model.build_and_train_model()
    app.run(host='localhost', port=8081, debug=True)
