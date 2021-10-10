from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
import os

import util

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        for filename in os.listdir('static/uploaded'):
            os.remove(os.path.join('static/uploaded', filename))
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        website = util.classify(uploaded_img_path)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               result=website)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0")
