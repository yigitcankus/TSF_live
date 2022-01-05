from app.paths import csvName
from app.models import User,Graph
import os
from app import app, db
from flask import render_template, url_for, flash, redirect, request
from app.forms import LoginForm, RegistrationForm, UploadForm
from flask_login import login_user, current_user, logout_user, login_required
import csv


@app.route("/deneme",methods=['GET', 'POST'])
def deneme():
    return "trial"