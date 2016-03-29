import flask
from flask import (Flask,
                  render_template,
                  request,
                  jsonify,
                  url_for
                  )

import binascii

from werkzeug import secure_filename
from shutil import copyfile

from kenkensolver import *

#Set the dummy to the default blank image

UPLOAD_FOLDER = '/upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

basedir = os.path.abspath(os.path.dirname(__file__))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS

def dated_url_for(endpoint, **values):
    if endpoint == 'js_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     'static/js', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    if endpoint == 'css_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     'static/css', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/')
def load():
    """
    Homepage: serve our visualization page, index.html
    """
    return render_template('index.html')

@app.route('/uploadajax', methods=["GET", "POST"])
def upldfile():
    if request.method == 'POST':
        files = request.files['file']
        print(files.filename)
        print('bad dog')
        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            print(filename)
            updir = os.path.join(basedir, 'upload/')
            files.save(os.path.join(updir, filename))
            file_size = os.path.getsize(os.path.join(updir, filename))
        else:
            app.logger.info('ext name error')
            return jsonify(error='ext name error')

        result = execute_solver(files)

        if result == 'good':
            with open("static/images/solution.jpg", "rb") as image_file:
                encoded_string = binascii.b2a_base64(image_file.read())
            return encoded_string

        elif result == 'error':
            print(result)
            with open("static/images/error.jpg", "rb") as image_file:
                encoded_string = binascii.b2a_base64(image_file.read())
            return encoded_string


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
