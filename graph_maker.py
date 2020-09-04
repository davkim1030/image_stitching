import argparse
import os
import cv2
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly
from PIL import Image


def resize(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    if height * width <= 2 ** 21:
        return img
    i = 2
    t_height, t_width = height, width
    while t_height * t_width > 2 ** 21:
        t_height = int(t_height / math.sqrt(i))
        t_width = int(t_width / math.sqrt(i))
        i += 1
    height, width = t_height, t_width
    image = Image.open(filename)
    resize_image = image.resize((height, width))
    filename = filename[:-1 * (len(filename.split(".")[-1]) + 1)] + "_resized." + filename.split(".")[-1]
    resize_image.save(filename)
    img = cv2.imread(filename)
    os.system("del " + filename)
    return img


def set_range(h, s, v):
    if 72 >= h >= 66 and 91 >= s >= 59 and 255 >= v >= 66:
        return 3
    elif 75 >= h >= 73 and 234 >= s >= 31 and 118 >= v >= 0:
        return 2
    elif 134 >= h >= 35 and 59 >= s >= 0 and 100 >= v >= 73:
        return 1
    return 0


def float_range(start, end, cnt):
    list = []
    step = (end - start) / (cnt - 1)
    for i in range(cnt):
        list.append(start + i * step)
    return list


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to input file name of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output image")
ap.add_argument("-lat1", "--latitude1", type=float, required=False,
                help="image's start latitude")
ap.add_argument("-lat2", "--latitude2", type=float, required=False,
                help="image's end latitude")
ap.add_argument("-lon1", "--longitude1", type=float, required=False,
                help="image's start longitude")
ap.add_argument("-lon2", "--longitude2", type=float, required=False,
                help="image's end longitude")
args = vars(ap.parse_args())

if (not args['latitude1'] and args['latitude2']) or (args['latitude1'] and not args['latitude2']):
    raise ValueError("latitude1 and latitude2 should be given together")
if (not args['longitude1'] and args['longitude2']) or (args['longitude1'] and not args['longitude2']):
    raise ValueError("longitude1 and longitude2 should be given together")

# Image Data Read
filename = args['image']
print("Image loading start")
print("Image resizing start")
img_color = resize(filename)
height, width = img_color.shape[:2]
print("Image resizing end")
print("Image loading end")

# resize image

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
print("Extracting HSV data start")
data = pd.DataFrame(np.concatenate(img_hsv))
print("Extracting HSV data end")
data.columns = ["H", "S", "V"]
print("Calculating \"Range\" data start")
data["Range"] = data.apply(lambda x: set_range(x['H'], x['S'], x['V']), axis=1)
data["Range"] = np.flip(data["Range"])
print("Calculating \"Range\" data end")

# Data Concentration Step
y = np.array(data['Range'])
z_data = pd.DataFrame(y.reshape(height, width))  # Size variations according to pixel
if args['longitude1']:
    z_data.index = float_range(args['longitude1'], args['longitude2'], height)
if args['latitude1']:
    z_data.columns = float_range(args['latitude1'], args['latitude2'], width)

# Create Heat Map
fig_heatmap = go.Figure(data=go.Heatmap(z=z_data,
                                        connectgaps=True,
                                        colorscale=[[0.0, "rgb(0,8,135)"],
                                                    [1.0, "rgb(0,135,8)"]]))
fig_heatmap.update_layout(title='HSV Graph', autosize=True,
                          margin=dict(l=65, r=50, b=65, t=90))

# Create 3D Graph
fig_3d = go.Figure(data=[go.Surface(z=z_data.values,
                                    colorscale=[[0.0, "rgb(0,8,135)"],
                                                [1.0, "rgb(0,135,8)"]])])
fig_3d.update_layout(title='HSV Graph', autosize=True,
                     scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                     margin=dict(l=65, r=50, b=65, t=90))

# Save and show graph
output = args['output']
if '.' in args['output']:
    output = args['output'][:-1 * (len(args['output'].split(".")[-1]) + 1)]
else:
    args['output'] = output + ".png"
plotly.io.write_image(fig_heatmap, output + "_heatmap." + args['output'].split(".")[-1])
plotly.io.write_image(fig_3d, output + "_3D_graph." + args['output'].split(".")[-1])
fig_heatmap.show()
fig_3d.show()
