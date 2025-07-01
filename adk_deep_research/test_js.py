from playwright.sync_api import sync_playwright
import time

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
    
    # Parameters to pass from Python
    amplitude = 2.0
    frequency = 0.5
    x_range = list(range(-10, 11))  # x values: [-10, -9, ..., 9, 10]
    
    # Execute JavaScript with parameters for calculation and plotting
    js_code = """
    (params) => {
        const { amplitude, frequency, xRange } = params;
        // Calculate y = amplitude * sin(frequency * x) using math.js
        const yValues = xRange.map(x => math.evaluate(`${amplitude} * sin(${frequency} * ${x})`));
        // Create Plotly chart
        return Plotly.newPlot('myDiv', [{
            x: xRange,
            y: yValues,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Sine Wave'
        }], {
            title: 'Sine Wave Plot',
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' }
        }).then(() => {
            // Return x and y values
            return { x: xRange, y: yValues };
        });
    }
    """
    
    # Pass parameters to JavaScript and execute
    result = page.evaluate(js_code, {
        "amplitude": amplitude,
        "frequency": frequency,
        "xRange": x_range
    })
    
    # Print the result returned from JavaScript
    print("Returned data:", result)
    
    # Keep the browser open for 10 seconds (or adjust as needed)
    print("Browser is open. Close it manually or wait 10 seconds.")
    time.sleep(100)
    
    # Close the browser
    browser.close()