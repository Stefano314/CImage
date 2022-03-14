# Image Processing Operations

This directory is a general archive of **images operations** that have been implemented using **python**.

Thresholding
------------
In the *Thresholding directory* there are generic thresholding operations. In particular, one very interesting one developed recently by a group of researchers[[1]](#1). This new technique proves to be both fast and accurate when compared to the other thresholding methods, as shown in the code presented in this repository.

Here are listed the outcomes of the thresholds on a test image:
### Thresholding Results
| Original | Global | Bersen | Niblack | Sauvola | New Technique |
| :---: |  :----:  | :---: | :---: | :---: | :---: |
| <img src="https://user-images.githubusercontent.com/79590448/144745064-53ec29ba-a53b-4b7c-9f43-1c39dac82f57.png" width="550">      | <img src="https://user-images.githubusercontent.com/79590448/144745077-b5e1fcf9-7773-448c-b4e1-ed5b10e32643.png" width="550">       | <img src="https://user-images.githubusercontent.com/79590448/144745087-c91a48e5-6d6f-45c8-af0e-e19cc61bdbe2.png" width="550">   | <img src="https://user-images.githubusercontent.com/79590448/144745100-23737164-b91e-455b-ad0c-4dd905185504.png" width="550"> | <img src="https://user-images.githubusercontent.com/79590448/144745130-4560620e-457f-430b-a8aa-33ae268483e7.png" width="550"> | <img src="https://user-images.githubusercontent.com/79590448/144745145-e11fe111-2e2a-4f39-a6bf-b61c3bdc48a0.png" width="550"> |

# CImage
This directory is an fancy and naive idea about creating a  **C++ library that manages images**, since it is extremely faster than python (In addition, it is fun). <br />
At the moment, the images are stored in a created class **cImage** and they must be **8-bit greyscale** images. <br /><br />
The methods available are:
- ```GetResolution()```: It shows the image resolutions.
- ```SummedAreaTable()```: It creates a 2-D vector containing the *Summed-area table*.
- ```LocalIntensity()```: It gives the value of the local intensity within a given window of an image.
- ```singh_threshold()```: This is the **threshold method** presented by the cited research groups[[1]](#1).
-  ```SaveImage()```: Saves the image stored in the cImage object.

Usage
-----
The idea is to create the *cImage* object and then simply call its methods:
```c++
int main() {
  string filename("ImageTest"); // Image name
  cImage fig(filename); // Load image
  cImage trans_image = fig.singh_threshold(11, 0.3); // Perform an operation
  trans_image.SaveImage("trans_image"); // Save the result
	return 0;
}
```

References
----------
<a id="1">[1]</a> 
T. Romen Singh et al. *"A New Local Adaptive Thresholding Technique in Binarization"*, IJCSI International Journal of Computer Science Issues, Vol. 8, Issue 6, No 2, November 2011.
