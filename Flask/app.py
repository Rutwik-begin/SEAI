from flask import Flask, render_template, request
from markupsafe import Markup
from triton_client import predict_image
from llm_client import generate_disease_info
import utils

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            img = file.read()
            prediction = predict_image(img)
            try:
                llm_response = generate_disease_info(prediction)

            except Exception as e:
                app.logger.error(f"Gemini Error: {str(e)}")

                llm_response = f"""
                <h2>Prediction</h2>
                <p>{prediction}</p>

                <h3>AI explanation temporarily unavailable.</h3>
                """
            app.logger.info(f"Prediction: {prediction}")
            res = Markup(llm_response.replace("\n", "<br>"))
            return render_template('display.html', status=200, result=res)
        except Exception as e:
            app.logger.error(str(e))
            return render_template(
                'index.html',
                status=500,
                res="Internal Server Error"
            )
    return render_template('index.html', status=500, res="Internal Server Error")


if __name__ == "__main__":
    app.run(
    host="0.0.0.0",
    port=5000,
    debug=True
    )
