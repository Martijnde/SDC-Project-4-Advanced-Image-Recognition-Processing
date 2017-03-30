# SDC-Project-4-Advanced-Image-Recognition-Processing

I also featured this project in a LinkedIn message! (for personal + Udacity branding)

https://www.linkedin.com/pulse/udacity-self-driving-car-engineering-project-4-advanced-de-boer?trk=v-feed&lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BcIIFwiOjkbC6rTj0Y0pwEQ%3D%3D

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###README 

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell called "Image Distortion on the chessboard images" of the IPython notebook* located in this repository, named: "Self Driving Car P4 Advanced Finding Lane lines.ipynb". 
(* All Notebook cells I am referring to in this README.md are also included in this ipynb file) 

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

 def warp(img, src_points, dst_points):
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped   
    
This resulted in the following source and destination points:
    
src_points = [(215, 715), (1120, 715), (606, 431), (676, 431)]

dst_points = [(200, 720), (1080, 720), (200, -500), (1080, -500)]

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/chessboard_output.JPG?raw=true)

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/straight_lines1.jpg?raw=true)

This resulted in this output:
![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/Image_Distortion_output.JPG?raw=true)

####2. Like shown in the notebook we plotted an example of one of the test images we used a combination of Thresholding (Sobel) functions after Image Distortion on to create a thresholded binary image. Here's an example of my output for this step:

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/Binary_test_image_output.png?raw=true)

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/Binary%20Warped%20images.JPG?raw=true)

This is a plot of the binary image lines to identify the lines:

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/plot_binary_image.JPG?raw=true)

binary plot 2:

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/Binary%20view.JPG?raw=true)


####3. The code for my perspective transform includes a function called `warp()`, which appears in the IPython notebook. The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

Code to define the points on the line:

img = warped_im
plt.imshow(img)

> src_points = [(290, 660), (1020, 660), (595, 450), (690, 450)]

> dst_points = [(200, 720), (1080, 720), (200, -500), (1080, -500)]

Plot of the points on the line:

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/line%20points.png?raw=true)

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/warped.png


####4. I identified lane-line pixels and fit their positions with a Sliding Windows function and Fitted a Polynomial! 

The steps I took are in the Notebook and also visable in the next two pictures:

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/color_fit_lines_new.jpg?raw=true)

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/color_fit_lines_newer.jpg?raw=true)


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To determine the curve of the lane lines I've done all the steps described in the lessons. With the polynomial fit for the left and right lane lines, I thereafter calculated the radius of curvature for each line according to formulas. I converted the distance units from pixels to meters, assuming 30 meters per 720 pixels in the vertical direction, and 3.7 meters per 700 pixels in the horizontal direction. For the final radius of curvature I took the average of both lines.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the included notebook file `Self_Driving_Car_P4_Advanced_Finding_Lane_Lines.ipynb` in the function `def process_video_pipeline(image):`.  

Here is an example of my result on a test image:

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/pipeline_output.png?raw=true)

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my output video result on youtube: 

https://www.youtube.com/watch?v=mUxlQXEE9PM

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took was just follow the lessons and be good, so I took the steps featured above, the pipeline might fail in the part of the video with shadow on the road, (like three shadow for example) or when we face bad lane lines for example, I might improve it if I were going to pursue this project further by tweaking the color settings to rule this out.  

The techniques I used were:

* Remove outliers before fitting the polynomials.
* Tweak thresholds
* Compare frame with previous frame(s) to reduce sudden changes
* Implement region of interest (ROI)

I found it very hard to round the lines of the curvature + printing them, this was also the case for the offset numbers.

### Thanks and credits:

I want to thank and add credits to the following great classmates for their overall help and support: Indra den Bakker, Jeroen Walta, Mikel Bobar-Irazar, Carlos Fernandes and Kyle Stewart-Frantz!!

What's next?

My aim is to combine this pipeline with a new one I have to build for Project 5, the aim of the next project is to detect other vehicles on the road. I will do this by using deep learning to build a neural network to detect the other vehicles on the road next to the lane line finding!

###Fun

This code created the amazing image showed below, might be fun to see :)

image = mpimg.imread('test_images/test6.jpg')
thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
plt.plot(binary)
plt.show()

![alt tag](https://github.com/Martijnde/SDC-Project-4-Advanced-Image-Recognition-Processing/blob/master/Fun.JPG?raw=true)
