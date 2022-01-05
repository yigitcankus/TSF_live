from flask import Flask
from flask_sqlalchemy import SQLAlchemy
#from flask_bcrypt import Bcrypt
from flask_login import LoginManager


app = Flask(__name__)

app.config["SECRET_KEY"] = '3a3058b1e0d1c12f1675a6d2c73694c4'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"

db = SQLAlchemy(app)

#bcrypt = Bcyrpt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

from app import paths