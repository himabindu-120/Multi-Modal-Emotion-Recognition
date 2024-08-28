from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from werkzeug.utils import secure_filename
from website.views import views 

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

 # Import the Bid model

# Register the blueprint
app.register_blueprint(views)

# Directory where uploaded images will be saved
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/category/<category_name>')
def show_category(category_name):
    products = Product.query.filter_by(category_name=category_name).all()
    if not products:
        return "Category not found", 404
    return render_template('category.html', category_name=category_name, items=products)


@app.route('/show_bids')
def show_bids():
    bids = Bid.query.all()
    return render_template('show_bids.html', bids=bids)


if __name__ == '__main__':
    app.run(debug=True)
