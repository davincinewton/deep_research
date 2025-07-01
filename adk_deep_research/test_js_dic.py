from playwright.sync_api import sync_playwright
import time

# Define the Python dictionary to pass to JavaScript
params = {
    "coefficients": {"a": 1.0, "b": -2.0, "c": 1.0},  # For y = a*x^2 + b*x + c
    "xRange": list(range(-5, 6)),  # x values: [-5, -4, ..., 4, 5]
    "plotTitle": "Quadratic Function"
}

# Launch Playwright
with sync_playwright() as p:
    # Set up Chromium browser with headless=False to keep it open
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    
    # Navigate to a blank page
    page.goto("about:blank")
    
    # Inject math.js and Plotly.js from CDNs
    page.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/mathjs/10.6.4/math.min.js")
    page.add_script_tag(url="https://cdn.plot.ly/plotly-latest.min.js")
    
    # Wait for both libraries to load
    page.wait_for_function("typeof math !== 'undefined' && typeof Plotly !== 'undefined'")
    
    # Add a div for the chart with explicit dimensions
    page.set_content('<div id="myDiv" style="width: 100%; height: 400px;"></div>')
    
    # Wait for the div to be available
    page.wait_for_selector("#myDiv", state="attached")
    
    # JavaScript code to process the dictionary, compute, and plot
    js_code = """
    (params) => {
        const { coefficients, xRange, plotTitle } = params;
        const { a, b, c } = coefficients;
        // Calculate y = a*x^2 + b*x + c using math.js
        const yValues = xRange.map(x => math.evaluate(`${a} * ${x}^2 + ${b} * ${x} + ${c}`));
        // Create Plotly chart
        return Plotly.newPlot('myDiv', [{
            x: xRange,
            y: yValues,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Quadratic'
        }], {
            title: plotTitle,
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' }
        }).then(() => {
            // Return computed data
            return { x: xRange, y: yValues, title: plotTitle };
        });
    }
    """
    
    # Pass the Python dictionary to JavaScript and execute
    result = page.evaluate(js_code, params)
    
    # Print the returned dictionary
    print("Returned data:", result)
    
    # Keep the browser open for 10 seconds for inspection
    print("Browser is open. Close it manually or wait 10 seconds.")
    time.sleep(100)
    
    # Close the browser
    browser.close()