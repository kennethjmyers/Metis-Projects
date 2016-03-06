import flask
from flask import (Flask,
                  render_template,
                  request,
                  json,
                  jsonify,
                  send_from_directory,
                  url_for,
                  redirect)
import os
import json
from nlp_analysis import *

app = Flask(__name__, static_url_path='')

cluster_object = cluster_analysis()


@app.route('/')
def load():
    """
    Homepage: serve our visualization page, index.html
    """
    with open('static/anime_titles3.txt') as f:
        titles = f.read().split(',')

    return render_template('index.html', titles=titles)

@app.route('/data', methods=["GET", "POST"])
def get_inputs():
    data = flask.request.json
    print(data['cluster_option'])
    html1, json1 = cluster_object.getPlot(data['series'], data['rec_num'], data['cluster_option'])
    #print(html1)
    #print(type(html1))
    full_json = jsonify(html=html1, json=json1)

    return full_json


@app.route('/static')
def send():
    return "<a href={}>{}</a>".format(url_for('static', filename='anime_titles2.txt'), url_for('static', filename='anime_titles2.txt'))

@app.route('/static')
def serve_static(filename):
    return send_from_directory(os.path.join(root_dir, 'static'), 'anime_titles2.txt')



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
    url_for('static', filename='anime_titles2.txt')
