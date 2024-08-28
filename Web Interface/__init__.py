# website/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

# Initialize SQLAlchemy instance
db = SQLAlchemy()
DB_NAME = "database.db"

# Factory function to create Flask app
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'ghcvgvk hiuhilwediwe'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    # Import blueprints
    from .views import views
    from .auth import auth

    # Register blueprints with the app
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    # Initialize extensions
    

    # Import models and create tables
    from .models import User,Product 
    with app.app_context():
        db.create_all()

    # Initialize LoginManager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    # Define user loader callback
    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app

# Function to create database if it doesn't exist
def create_database(app):
    db.create_all(app=app)
    print('Created Database!')
