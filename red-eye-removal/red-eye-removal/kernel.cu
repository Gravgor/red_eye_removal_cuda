#include "cuda_runtime.h";
#include "device_launch_parameters.h";
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "modp_b64.h"

#define MAX_ENCODE_BUFFER_SIZE 7840

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s, line %d\n", cudaGetErrorString(error), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define STBI_CHECK(call) \
    do { \
        if (!(call)) { \
            fprintf(stderr, "STB image error: %s\n", stbi_failure_reason()); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)



__global__ void removeRedEye(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channel;

        unsigned char r = srcImage[idx];
        unsigned char g = srcImage[idx + 1];
        unsigned char b = srcImage[idx + 2];

        if (r > 150 && g < 100 && b < 100) {
			unsigned char r_avg = 0;
			unsigned char g_avg = 0;
			unsigned char b_avg = 0;
			int count = 0;

			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					int x_ = x + i;
					int y_ = y + j;

					if (x_ >= 0 && x_ < width && y_ >= 0 && y_ < height) {
						int idx_ = (y_ * width + x_) * channel;
						r_avg += srcImage[idx_];
						g_avg += srcImage[idx_ + 1];
						b_avg += srcImage[idx_ + 2];
						count++;
					}
				}
			}

			r_avg /= count;
			g_avg /= count;
			b_avg /= count;

			dstImage[idx] = r_avg;
			dstImage[idx + 1] = g_avg;
			dstImage[idx + 2] = b_avg;
		}
		else {
			dstImage[idx] = r;
			dstImage[idx + 1] = g;
			dstImage[idx + 2] = b;
		}
    }
}

int main()
{
    int width, height, channel;
    unsigned char* srcImage = stbi_load("red_eye.jpg", &width, &height, &channel, 0);
    STBI_CHECK(srcImage != NULL);

    unsigned char* dstImage = (unsigned char*)malloc(width * height * channel);
    if (dstImage == NULL) {
        fprintf(stderr, "Failed to allocate memory for destination image\n");
        exit(EXIT_FAILURE);
    }

    unsigned char* d_srcImage, * d_dstImage;
    CUDA_CHECK(cudaMalloc(&d_srcImage, width * height * channel));
    CUDA_CHECK(cudaMalloc(&d_dstImage, width * height * channel));

    CUDA_CHECK(cudaMemcpy(d_srcImage, srcImage, width * height * channel, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    removeRedEye << <grid, block >> > (d_srcImage, d_dstImage, width, height, channel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dstImage, d_dstImage, width * height * channel, cudaMemcpyDeviceToHost));

    stbi_write_jpg("red_eye_removed.jpg", width, height, channel, dstImage, 100);

    char* imageData = reinterpret_cast<char*>(dstImage);
    int encodeLength = modp_b64_encode_len(width * height * channel);
    char* base64Image = (char*)malloc(encodeLength);
    if (base64Image == NULL) {
        fprintf(stderr, "Failed to allocate memory for base64 encoding\n");
        exit(EXIT_FAILURE);
    }

    const char* base64ImageConst = base64Image;
    printf("base64ImageConst: %s\n", base64ImageConst);

    free(base64Image);
    stbi_image_free(srcImage);
    free(dstImage);

    CUDA_CHECK(cudaFree(d_srcImage));
    CUDA_CHECK(cudaFree(d_dstImage));

    return 0;
}