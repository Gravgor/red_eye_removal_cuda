#include "cuda_runtime.h";
#include "device_launch_parameters.h";
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


// This project is a red eye remover. It takes an image and removes the red eye from it with use of CUDA.

// The image is loaded and the red eye is detected by checking the red channel of the image. If the red channel is above a certain threshold, the pixel is marked as red eye.

// The red eye is then replaced with the average color of the surrounding pixels.

// The image is then saved as a new image.

// The image is loaded and saved using the stb_image library.

// The image is processed using CUDA.

__global__ void removeRedEye(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int red = srcImage[(y * width + x) * channel + 0];
        int green = srcImage[(y * width + x) * channel + 1];
        int blue = srcImage[(y * width + x) * channel + 2];

        if (red > 150 && red > green && red > blue)
        {
            int sumRed = 0;
            int sumGreen = 0;
            int sumBlue = 0;
            int count = 0;

            for (int i = -2; i <= 2; i++)
            {
                for (int j = -2; j <= 2; j++)
                {
                    if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height)
                    {
                        sumRed += srcImage[((y + j) * width + (x + i)) * channel + 0];
                        sumGreen += srcImage[((y + j) * width + (x + i)) * channel + 1];
                        sumBlue += srcImage[((y + j) * width + (x + i)) * channel + 2];
                        count++;
                    }
                }
            }


            dstImage[(y * width + x) * channel + 0] = sumRed / count;
            dstImage[(y * width + x) * channel + 1] = sumGreen / count;
            dstImage[(y * width + x) * channel + 2] = sumBlue / count;
        }
        else
        {
            dstImage[(y * width + x) * channel + 0] = red;
            dstImage[(y * width + x) * channel + 1] = green;
            dstImage[(y * width + x) * channel + 2] = blue;
        }
    }
}


int main()
{
	int width, height, channel;
	unsigned char* srcImage = stbi_load("red_eye.jpg", &width, &height, &channel, 0);

	unsigned char* dstImage = (unsigned char*)malloc(width * height * channel);

	unsigned char* d_srcImage;
	unsigned char* d_dstImage;

	cudaMalloc(&d_srcImage, width * height * channel);
	cudaMalloc(&d_dstImage, width * height * channel);

	cudaMemcpy(d_srcImage, srcImage, width * height * channel, cudaMemcpyHostToDevice);

	dim3 block(32, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	removeRedEye << <grid, block >> > (d_srcImage, d_dstImage, width, height, channel);

	cudaMemcpy(dstImage, d_dstImage, width * height * channel, cudaMemcpyDeviceToHost);

	stbi_write_jpg("red_eye_removed.jpg", width, height, channel, dstImage, 100);

	stbi_image_free(srcImage);
	free(dstImage);

	cudaFree(d_srcImage);
	cudaFree(d_dstImage);

	return 0;
}