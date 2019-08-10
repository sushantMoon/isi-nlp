from flask import Flask, render_template, g, request, jsonify
from query_prediction_lstm import lstm
import os
import tensorflow as tf


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
      predictions = lstm_model.get_prediction(message=cleaned)
   print("Predictions : {}".format(predictions))
   return jsonify(predictions)

@app.route('/')
def index_page():
   return render_template("index.html")

if __name__ == '__main__':
   global lstm_model, graph
   lstm_model = lstm()
   graph = tf.get_default_graph()
   app.run(host='localhost', port=8081, debug=True)