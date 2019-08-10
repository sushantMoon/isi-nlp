from flask import Flask, render_template, request
from character_frequency import character_frequency
app = Flask(
   __name__,
   static_url_path='', 
   static_folder='templates',
   template_folder='templates'
   )

@app.route('/')
def index_page():
   return render_template("index.html",result = {})

@app.route('/',methods = ['POST', 'GET'])
def result():
    cf = character_frequency()
    string = request.form.get('assignment0-text-area')
    char_dict = cf.character_frequency(string)
    return render_template("index.html",result = char_dict)

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)

