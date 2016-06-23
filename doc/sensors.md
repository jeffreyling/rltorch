 # Sensors #

Different sensors are provided

# BatchVectorSensor

It transforms a `(n)` torch Tensor to a `(1,n)` tensor. 

# BatchVectorSensor_ForAtari

It is a special sensor for the Atari environment. It:
* rescales the Atari images to `width,height`
* it (potentially) transform the image to a grayscale image
* the final image is provided to the policy as a `(1,n)` vector

NOTE: This sensor cannot be used in openai Gym since the Atari environment in openai gym does not use the same observation format than the Atari environment used in this package

# IdSensor

Just the identity (may be useful)

# TilingSensor2D

Takes a tensor with two values, and returns a `1 , s1 x s2` tensor corresponding to a discretization of the first and second value (Tiling). A more generic version is under development...

