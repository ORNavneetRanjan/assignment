from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/assignment1')
def assignment1():
    return render_template('assignment1.html')

@app.route('/assignment2')
def assignment2():
    return render_template('assignment2.html')

@app.route('/assignment3')
def assignment3():
    return render_template('assignment3.html')

if __name__ == '__main__':
    app.run(debug=True)