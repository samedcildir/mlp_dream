# Artificial Brain Dream

I started this project for understanding how mlp classifier behaves under different activation functions, for that I created an mlp with 2 input (x,y coordinates) and 1 output (2 classes / 1 and 0) and printed each pixel's value in a Black/White image. I trained the image with random datas. After that I moved these datas slowly to see how mlp responds to changing inputs.

After completing this task, I realized that the resulting images and animations actually looks good so I started adding colour in RGB and HSV, added different randomness methods for choosing training pixels, added taking pixels from an image to see if how close the images become to the input image etc..

Since I added new features slowly and without any code structure, the code became dirty and since this is just a fun project I didn't clean it up fully so the code may seem bad and it may have bugs (especially with memory allocation however I think it doesn't have memory leak but it has "still reachable" heap memory when it ends but its not a problem since OS cleans them for us).

All of the available configurations are in the "configurations.h" file. It should be self explaining and it should give errors with static_asserts when conflicting configurations are made.

In order to make this image generation real-time, I used GPU (CUDA) and multithreaded architecture.
