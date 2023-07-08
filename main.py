import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from skimage import io
from scipy.signal import find_peaks
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, morphology
from skimage.measure import label
from matplotlib.patches import Circle
import io
import base64
import dash_canvas
import dash_table

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H1("JWST Image Analysis"), className="text-center mt-5")
        ),
        dbc.Row(dbc.Col(dcc.Upload(
                id="upload-image",
                children=html.Div(
                    ["Drag and drop or click to select an image of JWST"]
                ),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
                multiple=False,
            ),
            width=12,
        )),
        dbc.Row(dbc.Col(
            dcc.Loading(
                dcc.Graph(id="spectrum-graph"),
                type="default",
                style={"height": "300px"},
            ),
            width=12,
        )),
        html.Div(
            [
                html.H3("Original Image"),
                html.Div(id="original-image-container"),  # Placeholder for the original uploaded image
                html.Div(id="filtered-images"),  # Placeholder for the filtered images
            ],
            className="filtered-images-container",
            style={"margin-top": "30px"},
        ),
        html.Div(id="stars-table")  # Placeholder for the detected stars table
    ],
    fluid=True,
)


def process_image(image):
    # Convert the image to grayscale
    grayscale_image = np.mean(image, axis=2)

    # Perform spectrum analysis
    spectrum = np.mean(grayscale_image, axis=0)

    # Find peaks in the spectrum
    peaks, _ = find_peaks(spectrum, distance=30, prominence=5)

    return spectrum, peaks


def apply_filters(image):
    # Apply Gaussian filter
    image_gaussian = filters.gaussian(image, sigma=1)

    # Apply median filter
    image_median = filters.median(image)

    # Apply maximum filter
    image_max = ndimage.maximum_filter(image, size=3)

    return image_gaussian, image_median, image_max


@app.callback(
    Output("spectrum-graph", "figure"),
    Output("original-image-container", "children"), 
    Output("filtered-images", "children"),
    Output("stars-table", "children"),
    Input("upload-image", "contents"),
    prevent_initial_call=True,
)
def update_spectrum(contents):
    if contents is None:
        # Handle case when no file is uploaded
        return go.Figure(), html.Div(), html.Div(), html.Div()

    # Get the uploaded image
    _, content_string = contents.split(",")
    image = Image.open(io.BytesIO(base64.b64decode(content_string)))

    # Convert the image to numpy array
    image_array = np.array(image)
    image_array2 = np.array(image)

    # Process the image
    spectrum, peaks = process_image(image_array)

    threshold_value = np.mean(image_array) + 2 * np.std(image_array)  # set the threshold value
    image_max = np.copy(image_array)
    image_max[image_max < threshold_value] = 0  # zero out all values below the threshold
    labeled = label(image_max)  # label the connected components in the image

    # Update the is_local_maximum function with local_maxima function
    lm = morphology.local_maxima(image_array)
    coords = np.where(np.logical_and(lm, image_array > threshold_value))
    x1, y1 = coords[0], coords[1]
    v = image_array[(x1, y1)]
    lim = 0.7
    indices = np.where(v > lim)[0]

    # Handle the case when no local maxima are found
    if len(indices) == 0:
        print("No local maxima found.")
        detected_stars = []
    else:
        x2, y2 = x1[indices], y1[indices]
        detected_stars = list(zip(x2, y2))
        # print("Stars detected at positions:", detected_stars)

        # Add red circles to the image_array
        for star in detected_stars:
            x, y = star
            image_array[x, y] = [255, 0, 0]  # Red color

    # Create the spectrum plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(spectrum)), y=spectrum, name="Spectrum"))
    fig.add_trace(
        go.Scatter(
            x=peaks,
            y=spectrum[peaks],
            mode="markers",
            name="Peaks",
            marker=dict(color="red", size=8),
        )
    )
    fig.update_layout(
        title="Spectrum Analysis",
        xaxis_title="Pixel",
        yaxis_title="Intensity",
        template="plotly_white",
    )

    # Apply filters to the image
    image_gaussian, image_median, image_max = apply_filters(image_array)
    canvas_width = 500
    canvas_height = 250

    # Display the original uploaded image
    original_image = html.Div(
        [
            html.H4("Original Uploaded Image"),
            html.Img(
                src=image_to_data_url(image_array2),
                style={"max-width": "100%", "max-height": "75%"},
            ),
        ],
        className="img-zoom-container",
        style={
            "height": "650px",
            "width": "100%",
            "border": "1px solid gray",
            "padding": "10px",
            "text-align": "center",
        },
    )

    # Display the filtered images with zoom functionality
    filtered_images = html.Div(
        [
            html.H3("Filtered Images"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Image with Gaussian Filter"),
                                dash_canvas.DashCanvas(
                                    id="canvas-gaussian",
                                    width=canvas_width,
                                    height=canvas_height,
                                    image_content=image_to_data_url(image_gaussian),
                                ),
                            ],
                            className="img-zoom-container",
                            style={
                                "height": "650px",
                                "width": "650px",
                                "border": "1px solid gray",
                                "padding": "10px",
                            },
                        ),
                        width=6,
                        style={"margin-bottom": "80px", "padding-right": "5px"},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Image withMedian Filter"),
                                dash_canvas.DashCanvas(
                                    id="canvas-median",
                                    width=canvas_width,
                                    height=canvas_height,
                                    image_content=image_to_data_url(image_median),
                                ),
                            ],
                            className="img-zoom-container",
                            style={
                                "height": "650px",
                                "width": "650px",
                                "border": "1px solid gray",
                                "padding": "10px",
                            },
                        ),
                        width=6,
                        style={"margin-bottom": "80px", "padding-left": "5px"},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Image with Maximum Filter"),
                                dash_canvas.DashCanvas(
                                    id="canvas-max",
                                    width=canvas_width,
                                    height=canvas_height,
                                    image_content=image_to_data_url(image_max),
                                ),
                            ],
                            className="img-zoom-container",
                            style={
                                "height": "650px",
                                "width": "650px",
                                "border": "1px solid gray",
                                "padding": "10px",
                            },
                        ),
                        width=6,
                        style={"margin-bottom": "80px", "padding-right": "5px"},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Image with detected stars"),
                                dash_canvas.DashCanvas(
                                    id="canvas-detected-stars",
                                    width=canvas_width,
                                    height=canvas_height,
                                    image_content=image_to_data_url(image_array),
                                ),
                            ],
                            className="img-zoom-container",
                            style={
                                "height": "650px",
                                "width": "650px",
                                "border": "1px solid gray",
                                "padding": "10px",
                            },
                        ),
                        width=6,
                        style={"margin-bottom": "80px", "padding-left": "5px"},
                    ),
                ],
                style={"margin-left": "-5px", "margin-right": "-5px"},
            ),
        ],
    )

    # Create a data table for the detected stars
    stars_table = html.Div(
        [
            html.H3("Detected Stars"),
            dash_table.DataTable(
                id="stars-data-table",
                columns=[{"name": "Star ID", "id": "id"}, {"name": "Position", "id": "position"}],
                data=[{"id": i+1, "position": f"{star[0]}, {star[1]}"} for i, star in enumerate(detected_stars)],
                style_cell={'textAlign': 'center'},
                style_header={'fontWeight': 'bold'},
            ),
        ],
        className="stars-table-container",
    )

    return fig, original_image, filtered_images, stars_table


def image_to_data_url(image):
    img = Image.fromarray(image.astype(np.uint8))  # Convert to uint8 data type
    img = img.convert("RGB")  # Convert to RGB mode
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data_url = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data_url}"


if __name__ == "__main__":
    app.run_server(debug=True)
