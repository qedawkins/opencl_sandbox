#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
static std::vector<cl_uchar> readSPIRVFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    std::vector<cl_uchar> ret;
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return ret;
    }

    size_t filesize = 0;
    is.seekg(0, std::ios::end);
    filesize = (size_t)is.tellg();
    is.seekg(0, std::ios::beg);

    ret.reserve(filesize);
    ret.insert(
        ret.begin(),
        std::istreambuf_iterator<char>(is),
        std::istreambuf_iterator<char>() );

    return ret;
}
 
int main(void) {
    // Create the two input vectors
    int i;
    const int M = 2;
    const int K = 4;
    const int N = 3;
    float *A = (float*)malloc(sizeof(float)*M*K);
    float *B = (float*)malloc(sizeof(float)*K*N);
    for(i = 0; i < M*K; i++) {
        A[i] = 1.0 * (float)i;
    }
    for(i = 0; i < K*N; i++) {
        B[i] = 1.0;
    }
 
    std::vector<cl_uchar> il = readSPIRVFromFile("staticMatmul/matrixMultiplyStatic.spv");
 
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
            &device_id, &ret_num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
 
    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            M*K * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            K*N * sizeof(float), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            M*N * sizeof(float), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            M*K * sizeof(float), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            K*N * sizeof(float), B, 0, NULL, NULL);
 
    auto clCreateProgramWithILKHR_ = (clCreateProgramWithILKHR_fn)
        clGetExtensionFunctionAddressForPlatform(
            platform(),
            "clCreateProgramWithILKHR");

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithILKHR_(context, il.data(), il.size(), &ret);
    if (ret) {
        if (ret == CL_INVALID_CONTEXT) {
            std::cout << "Invalid context" << std::endl;
        } else if (ret == CL_INVALID_VALUE) {
            std::cout << "Invalid Value" << std::endl;
        } else if (ret == CL_OUT_OF_RESOURCES) {
            std::cout << "Out of resources" << std::endl;
        } else if (ret == CL_OUT_OF_HOST_MEMORY) {
            std::cout << "Out of host memory" << std::endl;
        } else if (ret == CL_SUCCESS) {
            std::cout << "Success?" << std::endl;
        } else if (ret == CL_INVALID_OPERATION)  {
            std::cout << "Invalid operation" << std::endl;
        } else {
            std::cout << ret << std::endl;
        }
        exit(1);
    }
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "forward_dispatch_0_matmul", &ret);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
 
    // Execute the OpenCL kernel on the list
    const int TS = 1;
    const size_t local[2] = { TS, TS };
    const size_t global[2] = { M, N };
    cl_event event = NULL;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global, local, 0, NULL, &event);
 
    ret = clWaitForEvents(1, &event);
    // Read the memory buffer C on the device to the local variable C
    float *C = (float*)malloc(sizeof(float)*M*N);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            M*N * sizeof(float), C, 0, NULL, NULL);
 
    // Display the result to the screen
    for(i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << std::endl;
        }
    }
 
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}
