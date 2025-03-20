# Python Image Processing Tutorial: Downloading Images and Performing Flip Operations 
 
In  this  tutorial,  we  will  learn  how  to  download  images  using  Python  and  perform  basic  image 
operations such as flipping and rotating using the Pillow library. 
 
## Prerequisites 
 
Before we begin, make sure you have the following libraries installed in your Python environment: 
 
- `requests`: for downloading images 
- `Pillow`: for image processing 
 
If you haven't installed these libraries yet, you can install them using pip: 
 
```bash 
pip install requests Pillow 
``` 
 
## Step 1: Downloading an Image 
 
First, we need to download an image. We will use the `requests` library to accomplish this task. 
 
``` 
import requests 
 
def download_image(url, filename): 
        response = requests.get(url) 
        if response.status_code == 200: 
                with open(filename, 'wb') as file: 
                        file.write(response.content) 
        else: 
                print(f"Error: Failed to download image from {url}") 
 
# Example usage 
image_url = "https://example.com/image.jpg"    # Replace with the URL of the image you want to 
download 
filename = "downloaded_image.jpg" 
download_image(image_url, filename) 
``` 
 
## Step 2: Opening and Displaying the Image 
 
Next, we will use the `Pillow` library to open and display the image we just downloaded. 
 
``` 
from PIL import Image 
 
def open_and_show_image(filename): 
        image = Image.open(filename) 
        image.show() 
 
# Example usage 
open_and_show_image(filename) 
``` 
 
## Step 3: Flipping and Rotating the Image 
 
Now we can perform flip and rotate operations on the image. The `Pillow` library provides several 
methods for image manipulation. 
 
``` 
def flip_image(filename, mode='horizontal'): 
        image = Image.open(filename) 
        if mode == 'horizontal': 
                flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT) 
        elif mode == 'vertical': 
                flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM) 
        else: 
                print("Error: Mode should be 'horizontal' or 'vertical'") 
                return 
        flipped_image.show() 
        return flipped_image 
 
def rotate_image(filename, degrees): 
        image = Image.open(filename) 
        rotated_image = image.rotate(degrees) 
        rotated_image.show() 
        return rotated_image 
 
# Example usage 
flipped_image = flip_image(filename, mode='horizontal')    # Horizontally flip 
flipped_image.save("flipped_horizontal.jpg")    # Save the horizontally flipped image 
 
flipped_image = flip_image(filename, mode='vertical')    # Vertically flip 
flipped_image.save("flipped_vertical.jpg")    # Save the vertically flipped image 
 
rotated_image = rotate_image(filename, 90)    # Rotate by 90 degrees 
rotated_image.save("rotated_90.jpg")    # Save the rotated image 
``` 
 
## Step 4: Saving the Modified Image 
 
In  the  examples  above,  we  have  seen  how  to  save  flipped  and  rotated  images.  You  can  use  the 
`save` method to save any modified image. 
 
``` 
# Save the image 
def save_image(image, filename): 
        image.save(filename) 
 
# Example usage 
save_image(flipped_image, "flipped_image.jpg") 
save_image(rotated_image, "rotated_image.jpg") 
``` 
 
By  now,  you  have  learned  how  to  download  images  using  Python  and  perform  basic  image 
operations using the Pillow library. You can extend these basics to implement more complex image 
processing functions as needed. 