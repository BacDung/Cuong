from flask import Flask, render_template, request, flash
import solve
from sklearn.preprocessing import StandardScaler
import joblib           #savemodel

# a=solve.train()
filename = 'digits_classifier.joblib.pkl'
    #         #Save model
    # _ = joblib.dump(clf, filename, compress=9)
clf = joblib.load(filename)
app = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

standard_to = StandardScaler()

@app.route("/test", methods=['GET','POST'])
def test():
    if request.method == 'POST':        
        sentence = str(request.form['input'])
        b=solve.test_a_sentence(clf,sentence)
        if (b==1):
            flash('"{}" được dự đoán là tích cực!'.format(sentence))
        else:
            flash('"{}" được dự đoán là tiêu cực!'.format(sentence))
        return render_template('test.html')
    else:
        return render_template('test.html')

@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        url = str(request.form['input'])
        b=solve.test_a_link_shopee(clf,url)
        flash("Số đánh giá tích cực: "+str(b[0]))
        flash("Số đánh giá tiêu cực: "+str(b[1]))
        flash("Tỉ lệ phản hồi đánh giá tích cực: " + str(round(b[2]*100, 2))+"%")
        return render_template('index.html')
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
