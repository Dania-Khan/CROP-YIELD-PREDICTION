from flask import Flask , render_template , request
import crop


app = Flask(__name__)

@app.route("/", methods =['POST'])
def hello():
    if request.method == "POST":
        ton=request.form["tonnes/hectare"]
        cropy=crop.crop_pred(ton)
        cr=crop
    return render_template("index.html",my_crop = cr)

'''
@app.route("/sub",methods=['POST'])
def submit():
    if request.method == "POST":
        name=request.form["username"] #dictionary
    return render_template("sub.html", n=name)
'''

if __name__ == "__main__":
    app.run(debug=True)