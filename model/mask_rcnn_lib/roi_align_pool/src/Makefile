all: roi_align_kernel.cu roi_align_kernel.h
	 nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52



clean:
	rm roi_align_kernel.cu.o

# nvcc compiler flags
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options


#/usr/local/cuda-9.1/bin/nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
#/usr/local/cuda-9.0/bin/nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52



# https://github.com/jwyang/faster-rcnn.pytorch
# https://github.com/ruotianluo/pytorch-faster-rcnn
# https://github.com/soeaver/Pytorch_Mask_RCNN

#https://github.com/longcw/RoIAlign.pytorch


#int roi_align_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
#                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output)
#{
#    int runtimeVersion, driverVersion = 0;
#    // Console log
#    cudaDriverGetVersion(&driverVersion);
#    cudaRuntimeGetVersion(&runtimeVersion);
#    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);

