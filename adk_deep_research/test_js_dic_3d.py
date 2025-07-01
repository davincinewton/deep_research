from playwright.sync_api import sync_playwright
import time

params = {
    "coefficients": {"a": 1.0, "b": 1.0, "c": 0.0},
    "xRange": list(range(-5, 6)),
    "yRange": list(range(-5, 6)),
    "plotTitle": "3D Quadratic Surface"
}

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Plotly Test</title>
    <style>
        body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
        #myDiv {
            width: 800px; /* Plotly layout width will match this */
            height: 600px; /* Plotly layout height will match this */
            border: 2px solid red; /* Visual debug */
            background-color: lightyellow; /* Visual debug */
        }
    </style>
</head>
<body>
    <div id="myDiv"></div>
</body>
</html>
"""

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    page.set_viewport_size({"width": 950, "height": 750}) # Ensure ample browser window
    page.set_content(html_content)

    # Using a specific, well-tested Plotly version
    page.add_script_tag(url="https://cdnjs.cloudflare.com/ajax/libs/mathjs/10.6.4/math.min.js")
    page.add_script_tag(url="https://cdn.plot.ly/plotly-2.20.0.min.js") # e.g., Plotly v2.20.0

    page.wait_for_function("typeof math !== 'undefined' && typeof Plotly !== 'undefined'")
    page.wait_for_selector("#myDiv", state="visible")

    # Brief pause, just in case of a rendering lag (very unlikely to be the root cause here)
    time.sleep(0.2)

    js_code = """
    (params) => {
        const { coefficients, xRange, yRange, plotTitle } = params;
        const { a, b, c } = coefficients;

        const zValues = yRange.map(y =>
            xRange.map(x => math.evaluate(`${a} * ${x}^2 + ${b} * ${y}^2 + ${c}`))
        );

        const layout = {
            title: {
                text: plotTitle,
                x: 0.5, xanchor: 'center', // Horizontal center
                y: 0.98, yanchor: 'top',   // Position title's top edge near the top of the container
                yref: 'container',         // Y position relative to the overall div container
                xref: 'container'          // X position relative to the overall div container
            },
            width: 800,  // Should match div width
            height: 600, // Should match div height
            autosize: false,
            margin: {
                l: 60,  // Left margin
                r: 150, // Right margin (for colorbar)
                b: 60,  // Bottom margin
                t: 40,  // Top margin (space below the title, for the plot scene)
                        // The title itself is positioned via yref:container, y:0.98
                pad: 4
            },
            scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' },
                // Explicitly define the domain for the 3D scene
                // This is [x_start, x_end], [y_start, y_end] as fractions
                // of the area remaining *after* layout.margin is applied.
                // y: [0, 0.9] means the scene will occupy the bottom 90% of the
                // vertical space allocated for the plot (after top/bottom margins),
                // leaving the top 10% of that plotting area empty (or for other annotations).
                // We want the scene to use most of the space defined by margins.
                // The title is outside this via yref:'container'.
                // Let's ensure the scene doesn't think it needs to be above the title.
                 domain: {
                     x: [0, 1], // Use full width available after l/r margins
                     y: [0, 1]  // Use full height available after t/b margins
                                // (t=40 is for space BETWEEN container-positioned title and scene top)
                 }
            }
        };

        const trace = {
            x: xRange,
            y: yRange,
            z: zValues,
            type: 'surface',
            colorscale: 'Viridis',
            colorbar: {
                title: 'Z Value',
                x: 1.03,           // Position its left anchor slightly into the right margin area
                                   // (relative to the scene's plot area)
                xanchor: 'left',
                y: 0.5,            // Vertically center colorbar relative to the scene's height
                yanchor: 'middle',
                len: 0.75,         // Length of colorbar as 75% of the scene's height
                thickness: 20,
                xpad: 10           // Pixels between plot scene and colorbar
            }
        };

        console.log('Plotly Layout being used:', JSON.parse(JSON.stringify(layout)));
        console.log('Div dimensions:', document.getElementById('myDiv').offsetWidth, document.getElementById('myDiv').offsetHeight);


        return Plotly.newPlot('myDiv', [trace], layout).then(() => {
            return { x: xRange, y: yRange, z: zValues, title: plotTitle };
        });
    }
    """

    try:
        # Listen for console messages from the page
        # page.on("console", lambda msg: print(f"Browser Console: {msg.text()}"))
        result = page.evaluate(js_code, params)
        print("Returned data from JS:", result)
    except Exception as e:
        print(f"JavaScript error: {e}")
        browser.close()
        exit()

    print("Inspect the plot. Check browser console for layout object and div dimensions. Close manually or wait 25s.")
    page.wait_for_timeout(25000)

    browser.close()