from . import db
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)  # Seller's name
    number = db.Column(db.String(255), nullable=False)  # Phone number
    category_name = db.Column(db.String(255), nullable=False)  # Product category
    mrp = db.Column(db.Float, nullable=False)  # Maximum retail price
    description = db.Column(db.Text, nullable=False)  # Product description
    image = db.Column(db.String(255), nullable=True)  #