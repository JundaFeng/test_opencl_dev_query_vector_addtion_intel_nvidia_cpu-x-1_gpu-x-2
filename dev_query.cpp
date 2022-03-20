#include <iostream>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <CL/cl.h>

using namespace std;

void DEBUG_INFO(std::string info){
   std::cout << std::endl << "\x1b[34m" << info << "\x1b[0m" << std::endl;
}

struct {
   cl_device_type type;
   const char *name;
   cl_uint count;
} devices[] =
        {
                {CL_DEVICE_TYPE_CPU,         "CL_DEVICE_TYPE_CPU",         0},
                {CL_DEVICE_TYPE_GPU,         "CL_DEVICE_TYPE_GPU",         0},
                {CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", 0},
                {CL_DEVICE_TYPE_DEFAULT,     "CL_DEVICE_TYPE_DEFAULT",     0},
                {CL_DEVICE_TYPE_ALL,         "CL_DEVICE_TYPE_ALL",         0},
        };
const int NUM_OF_DEVICE_TYPES = sizeof(devices) / sizeof(devices[0]);

#define OCLBASIC_PRINT_TEXT_PROPERTY(NAME)                                \
{                                                                   \
 size_t property_length = 0;                                           \
 err = clGetDeviceInfo(device,NAME,0,0,&property_length);              \
 char* property_value = new char[property_length];                     \
 err = clGetDeviceInfo(device,NAME,property_length,property_value,0);  \
 std::cout << " " << #NAME << ": " << property_value << std::endl; \
 delete[] property_value;                                              \
}




// OpenCL kernel. Each work item takes care of one element of c
const char* kernelSource = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
"\n";
void printInfo() {
   // Discover and initialize the platforms
   cl_int err = CL_SUCCESS;
   cl_uint num_of_entries = 3;
   cl_uint num_of_platforms;
   cl_uint num_of_devices;
   // get total number of available platforms
   err = clGetPlatformIDs(0, nullptr, &num_of_platforms);
   std::cout << "Number of available platforms: " << num_of_platforms << std::endl;

   cl_platform_id *platforms = new cl_platform_id[num_of_platforms];
   // get IDs for all platforms
   err = clGetPlatformIDs(num_of_platforms, platforms, nullptr);

// List all platforms
   vector<string> platform_names;
   for (cl_uint i = 0; i < num_of_platforms; ++i) {
// Get the length for the i-th platform name
      size_t platform_name_length = 0;
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &platform_name_length);

// Get the name itself for the i-th platform
      char *platform_name = new char[platform_name_length];
      err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_length, platform_name, nullptr);

// Output platform name
      std::cout << " [" << i << "] " << platform_name << std::endl;
      platform_names.emplace_back(platform_name);
      delete[] platform_name;
   }

   // Discover the number of devices which are provided for the selected platform.


   for (cl_uint j = 0; j < num_of_platforms; j++) {
      cl_platform_id platform = platforms[j]; //
      DEBUG_INFO("Number of devices available for each type in " + platform_names[j]);
      // iterate over all device types
      for (int i = 0; i < NUM_OF_DEVICE_TYPES; ++i) {
         err = clGetDeviceIDs(platform, devices[i].type, num_of_entries, nullptr, &devices[i].count);
         if (CL_DEVICE_NOT_FOUND == err) {
            devices[i].count = 0;
            err = CL_SUCCESS;
         }
         std::cout << " " << devices[i].name << ": " << devices[i].count << std::endl;
         // get useful capabilities information for each device
         DEBUG_INFO("Detailed information for " + string(devices[i].name));
         cl_uint cur_num_of_devices = devices[i].count;
         if (cur_num_of_devices == 0) {
            continue; // there is no devices of this type; move to the next type
         }
         // Retrieve a list of device IDs with type selected by i
         cl_device_id *devices_of_type = new cl_device_id[cur_num_of_devices];
         err = clGetDeviceIDs(platform, devices[i].type, cur_num_of_devices, devices_of_type, nullptr);

         // Iterate over all devices of the current device type
         for (cl_uint device_index = 0; device_index < cur_num_of_devices; ++device_index) {
            std::cout << "\n" << devices[i].name << "[" << device_index << "]\n";
            cl_device_id device = devices_of_type[device_index];
#define OCLBASIC_PRINT_NUMERIC_PROPERTY(NAME, TYPE)              \
               {                                                    \
               TYPE property_value;                                 \
               size_t property_length = 0;                          \
               err = clGetDeviceInfo(                               \
               device,                                              \
               NAME,                                                \
               sizeof(property_value),                              \
               &property_value,                                     \
               &property_length                                     \
               );                                                   \
               assert(property_length == sizeof(property_value));   \
               std::cout                                            \
               << " " << #NAME << ": "                           \
               << property_value << std::endl;                      \
               }

            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_NAME);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_AVAILABLE, cl_bool);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_VENDOR);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_PROFILE);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_VERSION);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DRIVER_VERSION);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_OPENCL_C_VERSION);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_ADDRESS_BITS, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_t);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_IMAGE_SUPPORT, cl_bool);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_EXTENSIONS);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint);
         }
         delete[] devices_of_type;
      }

   }
   delete[] platforms;
}
int main()
{
   printInfo();
   return 0;
}
