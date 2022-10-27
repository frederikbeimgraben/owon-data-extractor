#!/bin/python3
# ------------------------------------------------------------------
# [Frederik Beimgraben] OWON Oscilloscope Data Extractor
# Converts an image from an owon oscilloscope to a csv file 
# Supports X/Y and X,Y/t plots
# ------------------------------------------------------------------
# Uses the following libraries:
# - numpy
# - matplotlib
# - opencv
# - argparse
# - PIL
# - pytesseract
# ------------------------------------------------------------------
# Input parameters:
# -i: input file (.bmp or .png)
# -o: output file (.csv) (default: input file name with .csv extension)
# -m: mode (xy or xyt) (default: xy)
# -n: degree of the polynomial used for interpolation (default: 3)
# ------------------------------------------------------------------

from typing import Generator, Tuple, List
from matplotlib import patches, ticker
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import os
import pytesseract as tess
from PIL import Image as ImageModule
Image = ImageModule.Image

# Parse input arguments
parser = argparse.ArgumentParser(description='OWON Oscilloscope Data Extractor')
parser.add_argument('-i', '--input', help='Input file (.bmp or .png)', required=True)
parser.add_argument('-o', '--output', help='Output file (.csv) (default: input file name with .csv extension)', required=False, default=None)
parser.add_argument('-m', '--mode', help='Mode (xy or xyt) (default: xy)', required=False, default='xy')
parser.add_argument('-d', '--deg', help='Degree of the polynomial used for interpolation (default: 1)', required=False, type=int, default=1)
args = parser.parse_args()

# Check if input file exists
if not os.path.isfile(args.input):
    print("Input file does not exist!")
    exit()

# Check if input file is a .bmp or .png file
if not args.input.endswith(".bmp") and not args.input.endswith(".png"):
    print("Input file is not a .bmp or .png file!")
    exit()

# Check if output parameter is set
if args.output is None:
    args.output = args.input

# Check if mode parameter is set
if args.mode is None:
    args.mode = "xy"

# Check if mode parameter is valid
if args.mode != "xy" and args.mode != "xyt":
    print("Mode parameter is not valid!")
    exit()

# Functions
## Function to get the image from the input file name
def get_image(input_file: str) -> Image:
    # Read image
    img = ImageModule.open(input_file)

    # If the image is not in RGB mode, convert it
    if img.mode != "RGB":
        img = img.convert("RGB")
    assert img.size == (800, 600)

    # Return image
    return img

## Function to get the graph area from the image
def get_graph(img: Image) -> Image:
    # Crop the image to the graph area
    # (20, 19) -> (779, 518)
    graph = img.crop((20, 19, 780, 500))

    # Return the graph area
    return graph

## Function to get the scaling info area from the image (X/Y)
def get_scaling_xy(img: Image) -> Tuple[Image, Image]:
    # Get the two scaling info areas
    scaling_x = img.crop((17, 520, 180, 540))
    scaling_y = img.crop((17, 540, 180, 560))

    # Return the two scaling info areas
    return scaling_x, scaling_y

def get_scaling_xyt(img: Image) -> Tuple[Image, Image, Image, Image]:
    # Get the three scaling info areas
    scaling_x = img.crop((17, 520, 180, 540))
    scaling_y = img.crop((17, 540, 180, 560))
    offset_t = img.crop((577, 0, 660, 18))
    scaling_t = img.crop((350, 520, 410, 540))

    # Return the three scaling info areas
    return scaling_x, scaling_y, offset_t, scaling_t

MISTAKES = {
    ' ': '',
    'O': '0',
    'o': '0',
    'I': '1',
    'l': '1',
    '-': '',
    '\n': '',
    '%': '',
}

POSSIBLE_UNITS = {
    'mV': 1e-3,
    'uV': 1e-6,
    'nV': 1e-9,
    'mv': 1e-3,
    'uv': 1e-6,
    'nv': 1e-9,
    'ms': 1e-3,
    'us': 1e-6,
    'ns': 1e-9
}


## Function to extract the scaling info from the scaling info area (X/Y)
def extract_scaling_xy(image: Image) -> Tuple[float, float]:
    # Extract offset (divs) and scale (volts/div)
    # Divide in the middle of the image
    scaling_area = image.crop((0, 0, 80, 20))
    offset_area = image.crop((80, 0, 160, 20))
    # Apply OCR
    scaling_str: str = tess.image_to_string(scaling_area, config="--psm 7")
    offset_str: str = tess.image_to_string(offset_area, config="--psm 7")
    # Replace common OCR errors
    for mistake, replacement in MISTAKES.items():
        scaling_str = scaling_str.replace(mistake, replacement)
        offset_str = offset_str.replace(mistake, replacement)
    # Extract the unit
    unit = [(unit, mult) for unit, mult in POSSIBLE_UNITS.items() if unit in scaling_str]
    if len(unit) == 0:
        unit = 1.0
        unit_str = "V"
    else:
        unit_str = unit[0][0]
        unit = unit[0][1]
    scaling_str = scaling_str.replace(unit_str, '')
    # Convert to float
    scaling = float(scaling_str) * unit
    # Extract the offset
    offset = float(offset_str.replace('div', '')[:4]) * scaling
    # Return the offset and the scale
    return scaling, offset

def extract_t(image: Image) -> float:
    # Extract the offset only
    # Apply OCR
    offset_str: str = tess.image_to_string(image, config="--psm 7")
    # Replace common OCR errors
    for mistake, replacement in MISTAKES.items():
        offset_str = offset_str.replace(mistake, replacement)
    # Extract the unit
    unit = [(unit, mult) for unit, mult in POSSIBLE_UNITS.items() if unit in offset_str]
    if len(unit) == 0:
        unit = 1.0
        unit_str = "s"
    else:
        unit_str = unit[0][0]
        unit = unit[0][1]
    offset_str = offset_str.replace(unit_str, '')
    # Convert to float
    try:
        offset = float(offset_str.replace('div', '')[:4]) * unit
    except ValueError:
        print("Incompatible image for xyt!")
        exit(1)
    # Return the offset
    return offset

def extract_v(image: Image) -> float:
    # Extract the scale only
    # Apply OCR
    scale_str: str = tess.image_to_string(image, config="--psm 7")
    # Replace common OCR errors
    for mistake, replacement in MISTAKES.items():
        scale_str = scale_str.replace(mistake, replacement)
    # Extract the unit
    unit = [(unit, mult) for unit, mult in POSSIBLE_UNITS.items() if unit in scale_str]
    if len(unit) == 0:
        unit = 1.0
        unit_str = "V"
    else:
        unit_str = unit[0][0]
        unit = unit[0][1]
    scale_str = scale_str.replace(unit_str, '')
    # Convert to float
    try:
        scale = float(scale_str.replace('div', '')[:4]) * unit
    except ValueError:
        print("Incompatible image for xyt!")
        exit(1)
    # Return the scale
    return scale


def get_interpolation_function(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    # Get the interpolation function
    # Get the average for each x value
    x_unique, x_unique_indices, x_unique_inverse, x_unique_counts = np.unique(x, return_index=True, return_inverse=True, return_counts=True)
    y_unique = np.zeros_like(x_unique)
    # Get the mean for each x value
    for i in range(len(x_unique)):
        y_unique[i] = np.mean(y[x_unique_inverse == i])
    # Interpolate using polyfit
    poly = np.polyfit(x_unique, y_unique, n)
    slope = None
    if n == 1:
        slope = poly[0]
    # Return the interpolation function
    return np.poly1d(poly, variable='x'), x_unique, y_unique, slope

SIZE_DIV = 50 # Size of the divs in pixels

## Function to get the data points from the graph area (X/Y)
## Units in divs
def get_data_xy(graph: Image) -> np.ndarray:
    # Find the following shape (1 = #e0e0e0, 0 = #000000)
    shape = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]) * 0xe0
    # Convert depth to 8-bit
    shape = shape.astype(np.uint8)
    # Convert to grayscale (copy)
    graph_gs = graph.convert("L")
    # Convert to numpy array
    graph_np = np.array(graph_gs)
    # Find the shape
    result = cv.matchTemplate(graph_np, shape, cv.TM_CCOEFF_NORMED)
    # Get the coordinates of the shape
    loc = np.where(result >= 0.9)
    # Get the coordinates of the center of the shape (1, 1)
    x = loc[1][1] + 1
    y = loc[0][1] + 1
    # Get the data points
    # Data points are 1 pixel wide and 1 pixel high and have
    # the color #00e0e0
    data = []
    # Convert to numpy array
    graph_np_rgb = np.array(graph)
    # Get all pixels with the color #00e0e0
    loc = np.where(np.all(graph_np_rgb == (0, 0xe0, 0xe0), axis=-1))
    # Comvert to a list of coordinates (relative to the center and in divs (float))
    return (y - loc[0]) / SIZE_DIV, (loc[1] - x) / SIZE_DIV

def get_data_xyt(graph: Image) -> Tuple[np.ndarray]:
    # Get the center at x // 2, y // 2
    x = graph.width // 2
    y = graph.height // 2
    # Get Channel 1 #e00000
    # Get Channel 2 #e0e000
    graph_np_rgb = np.array(graph)
    data_ch1_loc = np.where(np.all(graph_np_rgb == (0xe0, 0, 0), axis=-1))
    data_ch2_loc = np.where(np.all(graph_np_rgb == (0xe0, 0xe0, 0x00), axis=-1))
    # Return the data points
    return ((y - data_ch1_loc[0]) / SIZE_DIV, (data_ch1_loc[1] - x) / SIZE_DIV), ((y - data_ch2_loc[0]) / SIZE_DIV, (data_ch2_loc[1] - x) / SIZE_DIV)

def get_unique_xyt(
        x1: np.ndarray,
        x2: np.ndarray,
        ch1: np.ndarray,
        ch2: np.ndarray,
        scaling1: float=1,
        scaling2: float=1,
        offset1: float=0,
        offset2: float=0
    ) -> Tuple[np.ndarray]:
    # Get the unique x values
    x_unique = np.unique(np.concatenate((x1, x2)))
    # Extract the y values for each x value and save their mean
    y1_unique = np.array(
        [
            np.max(ch1[x1 == x] if x in x1 else ch2[x2 == x])
            for x in x_unique
        ]
    ) * scaling1 + offset1
    y2_unique = np.array(
        [
            np.max(ch2[x2 == x] if x in x2 else ch1[x1 == x])
            for x in x_unique
        ]
    ) * scaling2 + offset2
    # Return the unique x and y values
    return x_unique, y1_unique, y2_unique

# Load image and show the graph area
img = get_image(args.input)
graph = get_graph(img)

# Get the scaling info areas
if args.mode == 'xy':
    scaling_x_im, scaling_y_im = get_scaling_xy(img)
else:
    scaling_x_im, scaling_y_im, offset_t_im, scaling_t_im = get_scaling_xyt(img)

scaling_x, offset_x = extract_scaling_xy(scaling_x_im)
scaling_y, offset_y = extract_scaling_xy(scaling_y_im)

print(f"X: {scaling_x}V/div, {offset_x}V")
print(f"Y: {scaling_y}V/div, {offset_y}V")

# Get the data points
if args.mode == 'xy':
    data_y, data_x = get_data_xy(graph)
    # Scale the data points
    data_x = data_x * scaling_x + offset_x
    data_y = data_y * scaling_y + offset_y
else:
    # Get time domain scaling and offset
    offset_t = round(extract_t(offset_t_im), 9)
    scaling_t = round(extract_t(scaling_t_im), 9)
    print(f't: {scaling_t}s/div, {offset_t}s')
    # Get the data points
    data_ch1, data_ch2 = get_data_xyt(graph)
    ch1_x = data_ch1[1] * scaling_t + offset_t
    ch2_x = data_ch2[1] * scaling_t + offset_t
    # Get the unique values
    x_unique, y1_unique, y2_unique = get_unique_xyt(
        ch1_x, ch2_x, 
        data_ch1[0], data_ch2[0],
        scaling1=scaling_x, scaling2=scaling_y,
        offset1=offset_x, offset2=offset_y
    )
    # Get the trigger channel (Pixel (680, 530))
    trigger_ch = 1 if np.all(np.array(img)[530, 680] == (0xe0, 0x00, 0x00)) else 2
    # Get the trigger voltage (706, 522) -> (738, 536)
    trigger_v_im = img.crop((706, 522, 783, 536))
    trigger_v = extract_v(trigger_v_im)
    print(f'Trigger: CH{trigger_ch}, {trigger_v}V')


if args.mode == 'xy':
    # Get 1st order interpolation function
    interpolation, x_unique, y_unique, slope = get_interpolation_function(data_x, data_y, args.deg)

    Uf = None

    interpolation_i, x_unique_i, y_unique_i, slope_i = get_interpolation_function(
        data_x - data_y, data_y, args.deg
    )

    # If slope is not none, calculate the resistance
    if slope is not None:
        resistors_deviation = 0.05
        voltage_b = 5*scaling_x
        voltage_c = interpolation(voltage_b)
        voltage_component = voltage_b - voltage_c
        current = voltage_c / 10
        resistance = voltage_component / current
        resistance_deviation = resistors_deviation * resistance
        print(f'U (B): {voltage_b:.2f}V')
        print(f'U (C): {voltage_c*1000:.2f}mV')
        print(f"R: {resistance:.2f}±{resistance_deviation:.2f}Ω")
    else:
        # Its probably a diode, so calculate the forward voltage (I = 6mA)
        current = 6e-3
        # Get the point at which interpolation == 10 * current
        # Get the inverse of the interpolation function
        roots = (interpolation_i - current * 10).roots
        # Get the largest real root
        roots = [
            root for root in roots 
            if np.isreal(root) and root > 0 and root.real < max(x_unique)
        ]
        Uf = (min(roots) if len(roots) > 0 else None).real
        # Print the result
        if Uf is not None:
            print(f'U: {Uf:.2f}V @ {current*1000:.2f}mA')
        else:
            print('Uf: N/A')

    # Save the data points
    if args.output is not None:
        # Save the data points as csv
        with open(args.output + '.raw.csv', 'w') as f:
            f.write('x,y\n')
            for i in range(len(data_x)):
                f.write(f'{data_x[i]},{data_y[i]}\n')
        # Save the current data points as csv
        with open(args.output + '.i.csv', 'w') as f:
            f.write('x,y\n')
            for i in range(len(x_unique_i)):
                f.write(f'{x_unique_i[i]},{y_unique_i[i]}\n')
        # Save the interpolation data points as csv
        with open(args.output + '.i.ip.csv', 'w') as f:
            f.write('x,y\n')
            for i in range(len(x_unique)):
                f.write(f'{x_unique[i]},{y_unique[i]}\n')

    # Create two subplots beside each other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # The first plot shows the relation the two voltages
    ax1.plot(data_x, data_y * 1000, '.', label='Data',  markersize=1)
    ax1.plot(x_unique, interpolation(x_unique) * 1000, '--', label='Interpolation')
    ax1.set_xlabel('B (V)')
    ax1.set_ylabel('C (mV)')
    ax1.legend()
    # The second plot shows the relation between the voltage and the current
    ax2.plot(data_x - data_y, data_y * 100, '.', label='Data',  markersize=1)
    ax2.plot(x_unique_i, interpolation_i(x_unique_i) * 100, '--', label='Interpolation')
    ax2.set_xlabel('B - C (V)')
    ax2.set_ylabel('I (mA)')
    ax2.legend()
    # Make lines
    for c in range(-5, 6):
        ax1.axvline(c * scaling_x, color='gray', linestyle='--', linewidth=0.5)
        ax2.axvline(c * scaling_x, color='gray', linestyle='--', linewidth=0.5)

        ax1.axhline(c * scaling_y * 1000, color='gray', linestyle='--', linewidth=0.5)
        ax2.axhline(c * scaling_y * 100, color='gray', linestyle='--', linewidth=0.5)

    # Add lines at the origin (black)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.7)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.7)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.7)

    if Uf is not None:
        ax2.axvline(Uf, color='red', linestyle='-', linewidth=0.6)
        ax2.axhline(6, color='red', linestyle='-', linewidth=0.6)
        # Write text at the bottom edge of the axes where the line crosses
        ax2.text(Uf, 6, f' Uf = {Uf:.2f}V', ha='left', va='bottom', color='red', size=16)
        # Put a dot at the intersection
        ax2.plot(Uf, 6, 'o', color='red', markersize=5)
    else:
        # Plot two arrows:
        ## From (0,0) to (1,0)
        ## From (1,0) to (1,slope_i)
        ax2.arrow(0, 0, voltage_component, 0, fc='k', ec='k', color='red')
        ax2.arrow(voltage_component, 0, 0, current*1000, fc='k', ec='k', color='blue')
        # Put text below the x-axis arrow on the right (resistance)
        ax2.text(voltage_component / 2, -0.5, f' R = U÷I ≈ {resistance:.0f}Ω', ha='center', va='top', color='black', size=16)
        # Put text on the middle of the y-axis arrow (to the right) (current) (rotated 90 degrees)
        ax2.text(voltage_component, slope_i * 125, f' I ≈ {current*1000:.2f}mA', ha='right', va='center', color='blue', rotation=90, size=16)
        # Put text on the middle of the x-axis arrow (to the left) (voltage)
        ax2.text(voltage_component / 2, 0, f' U ≈ {voltage_component:.2f}V', ha='center', va='bottom', color='red', size=16)

    # Set the distance between the axis annotations to scaling_x or scaling_y
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(scaling_x))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(scaling_y * 1000))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(scaling_x))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(scaling_y * 100))

    # Show the plot
    plt.show()
else:
    # Create three subplots beside each other and 2 (1 additional row) below on the right
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 5))
    # The first plot shows the time domain data for both channels
    ax1.plot(x_unique * 1000, y1_unique * 1000, '-', label='CH1',  markersize=1, color='red')
    ax1.plot(x_unique * 1000, y2_unique * 1000, '-', label='CH2',  markersize=1, color='yellow')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (mV)')
    # Add a line fr the trigger
    trigger_color = 'red' if trigger_ch == 1 else 'yellow'
    ax1.axhline(trigger_v*1000, color=trigger_color, linestyle='--', linewidth=1)
    # Add a grey line at 0V
    ax1.axhline(0, color='grey', linestyle='-', linewidth=1)
    # Set background color to black
    ax1.set_facecolor('black')
    # Add dark grey lines with a distance of scaling_x and scaling_y
    # -7 -> 7 for x
    # -5 -> 5 for y
    for c in range(-8, 9):
        ax1.axvline(c * scaling_t * 1000 + offset_t * 1000, color='gray', linestyle='--', linewidth=0.5)
    for c in range(-5, 6):
        ax1.axhline(c * scaling_x * 1000, color='gray', linestyle='--', linewidth=0.5)
    # Set the annotations to scaling_x and scaling_y
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(scaling_t * 5000))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(scaling_x * 1000))
    ax1.legend()
    # Make two spectrograms using specgram
    ax2.specgram(
        y1_unique[::10],
        Fs=0.1 / (x_unique[1] - x_unique[0]),
        scale='dB'
    )
    ax3.specgram(
        y2_unique[::10],
        Fs=0.1 / (x_unique[1] - x_unique[0]),
        scale='dB'
    )
    ax2.set_xlabel('Channel 1')
    ax2.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Channel 2')
    ax3.set_ylabel('Frequency (Hz)')
    # Remove x-axis labels
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    # Plot the Phase shift of the spectrograms in ax5
    fftshift = np.fft.fftshift(np.fft.fftfreq(len(y1_unique[::100]), x_unique[1] - x_unique[0]))
    angle = np.angle(np.fft.fftshift(np.fft.fft(y1_unique[::100]))) - np.angle(np.fft.fftshift(np.fft.fft(y2_unique[::100])))
    angle = angle[fftshift > 0]
    fftshift = fftshift[fftshift > 0]
    ax5.plot(
        fftshift / 100,
        angle
    )
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Phase shift (rad)')
    # Make the last plot invisible
    ax6.axis('off')
    # Plot the relative voltage in ax4
    ax4.plot(y1_unique * 1000, y2_unique * 1000, '.', label='CH1/CH2',  markersize=0.5, color='white')
    ax4.set_xlabel('CH1 (mV)')
    ax4.set_ylabel('CH2 (mV)')
    # Add a cross at 0,0
    ax4.axvline(0, color='grey', linestyle='-', linewidth=0.7)
    ax4.axhline(0, color='grey', linestyle='-', linewidth=0.7)
    ax4.legend()
    # Set background color to black
    ax4.set_facecolor('black')
    # Show the plot
    plt.show()
