# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

I re-defined the functions as I thought it was a good excercise to familizarize myself with the functions and its parameters. It was also helpful to assign them any name I wanted and writed their docstrings, adding a few urls here and there to help.

I kept everything in a separate `p1_funcs.py` file to keep the notebook more neat, and for future uses if needed.

The pipeline cosists of the following steps in order:
0. Read Image: this was originally part of the pipeline itself but later changed as it will need to receive the images for the video processing.
1. Color Transform: this defaults to gray scale, but lets you input another conversion if needed (as I used it in the write section of the pipeline to write the images as RGB)
2. Gauss Blur: Used to reduce the noise in the images, main parameter is the kernel size.
3. Canny: Used to find the edges in the image, needs an upper threhsold (above which everything is a kernel) and a low threshold below which edge is considred, inbetween region is defined as edge if it connected with an edge above the high threshold.
4. Mask: This is used to select a region of the image. This is important as the road lines are only on the road. It takes the previously processed image and a set of 4 coordinated or vertices to indicate the region.
5. Hough Coord: is the output of the hough lines algorithm, the function already provided in the notebook was broken into tow (hough_coords and plot_lines) to insert an inbetween function called interpolate. This function returns sets of coordinates for lines found out of the Canny image.
6. Interpolate: This function to process the Hough Coord output, and find the equations to all the lines provided. This was what took the most time, as I tried different things 
    1.weighting the different lines by length and having a proportional sum for the x = my + b equation
    2. Limiting the slopes
    3. Taking previous lines parameters.
It limits the absolute value of the slope to be between 1 and 2, uses a y_min and y_max obtained from the vertices (to set the lines in that region), recieves the line_equation from a previous iteration. It assigns the values by taking the length of each line, and getting its m, b parameters, if this line is the longest then it is more confident and keeps its m, b parameters. When it finishes returns the coordinates for the lines and the line eq for a following iteration.
7. Plot lines: Plots the lines, varies the thickness.
8. Weighted image: this is the final image, combinaes the lines and the road image in one, can change the transparency of both.

Finally the pipeline can show, write or return the final image.

The part that I had most fun doing is the Ipwidget to see in real time the changes to the image based on the parameters in the notebook.


### 2. Identify potential shortcomings with your current pipeline

1. Needs to be able to change with the image (resolution) I adjusted the vertices manually for the Optional video. Maybe set this as a proportion of the image dimensions. That would however imply that the camera is set in the exact position and covers the same image but with higher resolution, so still some intervention is needed.
2. Needs more tweaking with the parameters. I did an exploratory run on the parameters suggested in the lectures and they worked ok, so I didn't venture a lot in finding the optimal ones, if there is such a thing.
3. Noise, there noise when drawing the image, that is they can change abruptly sometimes. a smoothing function or memory could work.


### 3. Suggest possible improvements to your pipeline

I mentioned most in the previous section. Perhaps try additional image filters that can highlight the road lines more. Use smoothing for the jumps in drawing the lines. Try some other kind of interpolation (Quadratic?) for the curves.
