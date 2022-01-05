from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from app.models import User

class RegistrationForm(FlaskForm):
    name= StringField("Name", validators=[DataRequired(),
    Length(min=2, max=20)])

    surname= StringField("Surname", validators=[DataRequired(),
    Length(min=2, max=20)])
    
    username= StringField("Username", validators=[DataRequired(),
    Length(min=2, max=20)])

    email= StringField("Email", validators=[DataRequired(),
    Email()])

    birthyear= IntegerField("Birth Year", validators=[DataRequired()])

    password = PasswordField('Password', validators=[DataRequired(), Length(min=5, max=10)])
   
    area= StringField("Search Area:", validators=[DataRequired()])

    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])
    
    submit = SubmitField('Sign Up')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError("Name already exists, choose another")
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError("There is already a user using that email")


class LoginForm(FlaskForm):
    email= StringField('Email',
                        validators=[DataRequired(), Email()])
    
    password = PasswordField('Password', validators=[DataRequired(), Length(min=5, max=10)])
    
    remember = BooleanField('Remember Me')
    
    submit = SubmitField('Login')

class UploadForm(FlaskForm):
    file = FileField("Upload Your File Here", validators=[FileAllowed(["csv"])])
    upload = SubmitField('Upload')

class GraphForm(FlaskForm):
    days = IntegerField("How many days you want to see?", validators=[DataRequired()])
    upload = SubmitField('Show Me')

class DownloadForm(FlaskForm):
    download = SubmitField('Download')
