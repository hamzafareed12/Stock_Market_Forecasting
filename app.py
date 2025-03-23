from flask import Flask, render_template, request
from model import main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get user input
            ticker = request.form['ticker']
            periods = int(request.form['periods'])
            model_name = request.form['model']  # Selected model
            
            # SARIMAX parameters (if applicable)
            if model_name == "sarimax":
                p = int(request.form['p'])
                d = int(request.form['d'])
                q = int(request.form['q'])
                P = int(request.form['P'])  # Seasonal AR order
                sarimax_params = (p, d, q, P)
            else:
                sarimax_params = None
            
            # Run the selected model
            forecast, plot_url, error_plot_url = main(ticker, periods, model_name, sarimax_params)
            
            # Render the results
            return render_template('index.html', 
                                 ticker=ticker, 
                                 forecast=forecast.to_dict('records'), 
                                 plot_url=plot_url,
                                 error_plot_url=error_plot_url,
                                 model=model_name)
        except Exception as e:
            # Render the error message
            return render_template('index.html', error=str(e))
    
    # Render the input form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)