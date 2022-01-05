from app import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id=db.Column(db.Integer, primary_key=True)
    name=db.Column(db.String(25), unique=True)
    surname=db.Column(db.String(50))
    username=db.Column(db.String(25))
    email=db.Column(db.String(100), unique=True, nullable =False)
    birthyear=db.Column(db.Integer)
    password=db.Column(db.String(25), nullable =False)
    area=db.Column(db.String(50))

    csv_file=db.Column(db.String(100), default="default.csv") 

    posts = db.relationship('Graph', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.name}', '{self.email}', '{self.csv_file}')"


class Graph(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_file=db.Column(db.String(100), default="default.jpg") 
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Graph('{self.name}', '{self.image_file}')"

