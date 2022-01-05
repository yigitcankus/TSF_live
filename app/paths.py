from pandas import read_csv

from app.Univariate_forecasting import graphVal
from app.models import User, Graph
import os
from app import app, db, Univariate_forecasting, multivariate_forecasting
from flask import render_template, url_for, flash, redirect, request, send_from_directory
from app.forms import LoginForm, RegistrationForm, UploadForm, GraphForm, DownloadForm
from flask_login import login_user, current_user, logout_user, login_required
import csv
import pandas as pd

from app.multivariate_forecasting import timeinterval

ROOT_DIR = os.path.dirname(os.path.abspath("uploaded"))

@app.route("/home")
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/contactus")
def contactus():
    return render_template("contactus.html", title="Contact Us")


@app.route("/about")
def about():
    return render_template("about.html", title="About")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(name=form.name.data, surname=form.surname.data, username=form.username.data, email=form.email.data,
                    birthyear=form.birthyear.data, password=form.password.data, area=form.area.data)
        db.session.add(user)
        db.session.commit()
        flash(f'Your account created successfully', 'success')
        return redirect(url_for('login'))
    return render_template('signin.html', title='Sign Up', form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and (user.password == form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_file(form_file):
    random_name = "rsndom_file_name"
    f_name, f_ext = os.path.splitext(form_file.filename)
    f_total = f_name + f_ext
    file_path = os.path.join(app.root_path, 'static/uploaded', f_total)
    form_file.save(file_path)
    return f_total


@app.route("/profile", methods=['GET', 'POST'])
@login_required
def profile():
    form = UploadForm()
    if form.validate_on_submit():
        if form.file.data:
            uploaded_file = save_file(form.file.data)
            current_user.csv_file = uploaded_file
            db.session.commit()
            flash('Your file has been loaded!', 'success')
        else:
            flash('You did not upload any file!', 'danger')
        return redirect(url_for('profile'))
    return render_template('profile.html', title='Profile',
                           form=form)


# profile_pic = url_for('static', filename='profile picture/anon.jpg')
# return render_template('profile.html', title='My Profile')

def download_file(date, rfr_data, svr_data, xgb_data, dtr_data):
    data = {'Date': date,
            'Random Forest Forecast': rfr_data,
            'Support Vector Regression': svr_data,
            'XGBoost': xgb_data,
            'Decision Tree Regressor': dtr_data}
    df = pd.DataFrame(data)
    df.to_csv(ROOT_DIR + "/app/static/downloads/data.csv", index=False)
    #return send_from_directory(ROOT_DIR+"/static/downloads/", "covid.csv")


@app.route("/graph", methods=['GET', 'POST'])
def graph():

    form2 = DownloadForm()

        #return redirect(url_for('graph'))

    svr_error_rate = 0
    rfr_error_rate = 0
    dtr_error_rate = 0
    xgb_error_rate = 0
    form = GraphForm()
    csv = read_csv(ROOT_DIR + "\\app\\static\\uploaded\\" + current_user.csv_file)
    if form.validate_on_submit():
        if form.days.data:
            n = form.days.data
            labels = timeinterval(current_user.csv_file, n)
            if len(csv.columns) == 2:
                rfr_data = Univariate_forecasting.forecast_algorithm("rfr", n, current_user.csv_file)
                svr_data = Univariate_forecasting.forecast_algorithm("svr", n, current_user.csv_file)
                dtr_data = Univariate_forecasting.forecast_algorithm("dtr", n, current_user.csv_file)
                xgb_data = Univariate_forecasting.forecast_algorithm("xgb", n, current_user.csv_file)
            elif len(csv.columns) > 2:
                rfr_data = multivariate_forecasting.forecast_algorithm("rfr", n, current_user.csv_file)
                svr_data = multivariate_forecasting.forecast_algorithm("svr", n, current_user.csv_file)
                dtr_data = multivariate_forecasting.forecast_algorithm("dtr", n, current_user.csv_file)
                xgb_data = multivariate_forecasting.forecast_algorithm("xgb", n, current_user.csv_file)

            flash('Your days has been changed', 'success')
        else:
            flash('You did not choose any days!', 'danger')
    else:
        def_num = 19
        labels = timeinterval(current_user.csv_file, def_num)
        if len(csv.columns) == 2:
            rfr_data, rfr_error_rate = Univariate_forecasting.forecast_algorithm("rfr", def_num, current_user.csv_file)
            svr_data, svr_error_rate = Univariate_forecasting.forecast_algorithm("svr", def_num, current_user.csv_file)
            dtr_data, dtr_error_rate = Univariate_forecasting.forecast_algorithm("dtr", def_num, current_user.csv_file)
            xgb_data, xgb_error_rate = Univariate_forecasting.forecast_algorithm("xgb", def_num, current_user.csv_file)
        elif len(csv.columns) > 2:
            rfr_data, rfr_error_rate = multivariate_forecasting.forecast_algorithm("rfr", def_num, current_user.csv_file)
            svr_data, svr_error_rate = multivariate_forecasting.forecast_algorithm("svr", def_num, current_user.csv_file)
            dtr_data, dtr_error_rate = multivariate_forecasting.forecast_algorithm("dtr", def_num, current_user.csv_file)
            xgb_data, xgb_error_rate = multivariate_forecasting.forecast_algorithm("xgb", def_num, current_user.csv_file)



    legend = "Predictions for " + current_user.csv_file
    hey = current_user.csv_file
    graphVal(hey)
    # print(hey + "path")

    """for row in data:
        labels.append(row[0])
        values.append(row[1])"""

    if form2.validate_on_submit():
        uploaded_file = download_file(labels,rfr_data,svr_data,xgb_data,dtr_data)


    return render_template('graph.html', title="My Graphs", form=form, svr_data=svr_data, rfr_data=rfr_data,
                           dtr_data=dtr_data, xgb_data=xgb_data, labels=labels, legend=legend,
                           rfr_error_rate=rfr_error_rate, svr_error_rate=svr_error_rate, dtr_error_rate=dtr_error_rate,
                           xgb_error_rate=xgb_error_rate, form2 = form2)


'''
def Download():
    form2 = DownloadForm()
    if form2.validate_on_submit():
        if form2.file.data:
            uploaded_file = download_file(form2.file.data)
        return redirect(url_for('graph'))
    return render_template('graph.html', title='My Graphs',
                           form=form2)
'''

