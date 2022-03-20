//
// Created by jundafeng on 3/20/22.
//


#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <CL/opencl.h>
#include "cxxtimer.hpp"
const char *getErrorString(cl_int error)
{
   switch(error){
      // run-time and JIT compiler errors
      case 0: return "CL_SUCCESS";
      case -1: return "CL_DEVICE_NOT_FOUND";
      case -2: return "CL_DEVICE_NOT_AVAILABLE";
      case -3: return "CL_COMPILER_NOT_AVAILABLE";
      case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case -5: return "CL_OUT_OF_RESOURCES";
      case -6: return "CL_OUT_OF_HOST_MEMORY";
      case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case -8: return "CL_MEM_COPY_OVERLAP";
      case -9: return "CL_IMAGE_FORMAT_MISMATCH";
      case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case -11: return "CL_BUILD_PROGRAM_FAILURE";
      case -12: return "CL_MAP_FAILURE";
      case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
      case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
      case -15: return "CL_COMPILE_PROGRAM_FAILURE";
      case -16: return "CL_LINKER_NOT_AVAILABLE";
      case -17: return "CL_LINK_PROGRAM_FAILURE";
      case -18: return "CL_DEVICE_PARTITION_FAILED";
      case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

         // compile-time errors
      case -30: return "CL_INVALID_VALUE";
      case -31: return "CL_INVALID_DEVICE_TYPE";
      case -32: return "CL_INVALID_PLATFORM";
      case -33: return "CL_INVALID_DEVICE";
      case -34: return "CL_INVALID_CONTEXT";
      case -35: return "CL_INVALID_QUEUE_PROPERTIES";
      case -36: return "CL_INVALID_COMMAND_QUEUE";
      case -37: return "CL_INVALID_HOST_PTR";
      case -38: return "CL_INVALID_MEM_OBJECT";
      case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case -40: return "CL_INVALID_IMAGE_SIZE";
      case -41: return "CL_INVALID_SAMPLER";
      case -42: return "CL_INVALID_BINARY";
      case -43: return "CL_INVALID_BUILD_OPTIONS";
      case -44: return "CL_INVALID_PROGRAM";
      case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
      case -46: return "CL_INVALID_KERNEL_NAME";
      case -47: return "CL_INVALID_KERNEL_DEFINITION";
      case -48: return "CL_INVALID_KERNEL";
      case -49: return "CL_INVALID_ARG_INDEX";
      case -50: return "CL_INVALID_ARG_VALUE";
      case -51: return "CL_INVALID_ARG_SIZE";
      case -52: return "CL_INVALID_KERNEL_ARGS";
      case -53: return "CL_INVALID_WORK_DIMENSION";
      case -54: return "CL_INVALID_WORK_GROUP_SIZE";
      case -55: return "CL_INVALID_WORK_ITEM_SIZE";
      case -56: return "CL_INVALID_GLOBAL_OFFSET";
      case -57: return "CL_INVALID_EVENT_WAIT_LIST";
      case -58: return "CL_INVALID_EVENT";
      case -59: return "CL_INVALID_OPERATION";
      case -60: return "CL_INVALID_GL_OBJECT";
      case -61: return "CL_INVALID_BUFFER_SIZE";
      case -62: return "CL_INVALID_MIP_LEVEL";
      case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
      case -64: return "CL_INVALID_PROPERTY";
      case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
      case -66: return "CL_INVALID_COMPILER_OPTIONS";
      case -67: return "CL_INVALID_LINKER_OPTIONS";
      case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

         // extension errors
      case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
      case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
      case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
      case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
      case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
      case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
      default: return "Unknown OpenCL error";
   }
}

// OpenCL kernel. Each work item takes care of one element of c

const char *kernelSource =                                       "\n" \
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" 
"__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n){                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"    }                                                           \n" \
"}                                                               \n" \
                                                                "\n" ;

struct{
   cl_device_type device_type;
   std::string device_type_name;
} platform_device_pair[]={
        {CL_DEVICE_TYPE_CPU, "Intel(R) CPU Runtime for OpenCL(TM) Applications"},
        {CL_DEVICE_TYPE_GPU, "Intel(R) OpenCL HD Graphics"},
        {CL_DEVICE_TYPE_GPU, "NVIDIA CUDA"},
};

int main( int argc, char* argv[] ) {
   // Length of vectors
   unsigned int n = 10000000;

   // Host input vectors
   float *h_a;
   float *h_b;
   // Host output vector
   float *h_c;

   // Size, in bytes, of each vector
   size_t bytes = n * sizeof(float);

   std::cout << "Number of bytes in Giga: " << 3*static_cast<float>(bytes)/pow(10,9) << std::endl;
   // Allocate memory for each vector on host
   h_a = (float *) malloc(bytes);
   h_b = (float *) malloc(bytes);
   h_c = (float *) malloc(bytes);

   // Initialize vectors on host
   int i;
   for (i = 0; i < n; i++) {
      h_a[i] = 1.0*i/n;
      h_b[i] = 1.0*i/n;
   }



   cl_context context;               // context
   cl_command_queue queue;           // command queue
   cl_program program;               // program
   cl_kernel kernel;                 // kernel
   cl_int err;
   // Bind to platform
   cl_uint num_pltfs, num_devs, num_entries = 3;
   std::vector<cl_platform_id> platforms(num_entries);        // OpenCL platform
   std::vector<cl_device_id> device_ids(num_entries);           // device ID
   err = clGetPlatformIDs(num_entries, &platforms[0], &num_pltfs);
   if (err != CL_SUCCESS) {
      std::cout << "Cannot get platform" << std::endl;
      return -1;
   }

   for (i = 0; i < num_pltfs; ++i) {
// Get the length for the i-th platform name
      size_t platform_name_length = 0;
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &platform_name_length);

// Get the name itself for the i-th platform
      char *platform_name = new char[platform_name_length];
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_length, platform_name, nullptr);

// Output platform name
      std::cout << " [" << i << "] " << platform_name << std::endl;
      delete[] platform_name;
   }

   for(cl_uint i_pltf=0; i_pltf<num_pltfs; i_pltf++){
      timer_start("Vector addition on " + platform_device_pair[i_pltf].device_type_name, 'm');
      // Get ID for the device
      err = clGetDeviceIDs(platforms[i_pltf], platform_device_pair[i_pltf].device_type, num_entries, &device_ids[0], &num_devs);
      if (err != CL_SUCCESS) {
         std::cout << "Cannot get device" << std::endl;
         switch (err) {
            case CL_INVALID_PLATFORM: std::cout << "CL_INVALID_PLATFORM" << std::endl;
            case CL_INVALID_DEVICE_TYPE: std::cout << "CL_INVALID_DEVICE_TYPE" << std::endl;
            case CL_INVALID_VALUE: std::cout << "CL_INVALID_VALUE" << std::endl;
            case CL_DEVICE_NOT_FOUND: std::cout << "CL_DEVICE_NOT_FOUND" << std::endl;
         }
         return -1;
      }

      cl_device_id device_id = device_ids[0];

      // Create a context
      context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
      if (err != CL_SUCCESS) {
         std::cout << "Create context failed" << std::endl;
         return -1;
      }
      // Create a command queue
      queue = clCreateCommandQueue(context, device_id, 0, &err);
      if (err != CL_SUCCESS) {
         std::cout << "Create command queue failed" << std::endl;
         return -1;
      }

      // Device input buffers
      cl_mem d_a;
      cl_mem d_b;
      // Device output buffer
      cl_mem d_c;

      // Create the input and output arrays in device memory for our calculation
      d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_a, nullptr);
      d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_b, nullptr);
      d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
      if (d_a == nullptr || d_b == nullptr || d_c == nullptr) {
         std::cout << "Create buffer failed" << std::endl;
         return -1;
      }

      size_t globalSize, localSize;
      // Number of work items in each local work group
      localSize = 8;

      // Number of total work items - localSize must be devisor
      globalSize = static_cast<size_t>(ceil(n / (float) localSize) * localSize);

      // Write our data set into the input array in device memory
      err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                 bytes, h_a, 0, nullptr, nullptr);
      err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                  bytes, h_b, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
         std::cout << "Enqueue Write Buffer failed" << std::endl;
         return -1;
      }

      // Create the compute program from the source buffer
      program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, nullptr, &err);
      if (program == nullptr) {
         std::cout << "Create program failed" << std::endl;
         return -1;
      }
      // Build the program executable
      clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
      if (err != CL_SUCCESS) {
         std::cout << "Build program failed" << std::endl;

         return -1;
      }

      // Create the compute kernel in the program we wish to run
      kernel = clCreateKernel(program, "vecAdd", &err);
      if (kernel == nullptr) {
         std::cout << "Create kernel failed" << std::endl;
         return -1;
      }


      // Set the arguments to our compute kernel
      err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
      err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
      err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
      if (err != CL_SUCCESS) {
         std::cout << "Set kernel arg failed" << std::endl;
         return -1;
      }

      // Execute the kernel over the entire range of the data set
      err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                   0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
         std::cout << "Run kernel failed" << std::endl;
         return -1;
      }

      // Wait for the command queue to get serviced before reading back results
      clFinish(queue);

      // Read the results from the device
      clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
         std::cout << "Read data failed" << std::endl;
         return -1;
      }

      //Sum up vector c and print result divided by n, this should equal 1 within error
      float sum = 0;
      for (i = 0; i < n; i++)
         sum += h_c[i];
      std::cout << "Result on " + platform_device_pair[i_pltf].device_type_name + ": " << sum << std::endl;
      // release OpenCL resources
      clReleaseMemObject(d_a);
      clReleaseMemObject(d_b);
      clReleaseMemObject(d_c);
      clReleaseProgram(program);
      clReleaseKernel(kernel);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
      timer_stop('m');
   }

   //release host memory
   free(h_a);
   free(h_b);
   free(h_c);


   return 0;
}