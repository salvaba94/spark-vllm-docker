// Test E2M1 PTX instruction availability across architecture flags.
// Compile with: nvcc -arch=<arch> test_e2m1_flags.cu -o test_e2m1_flags
// Run on SM120/SM121 hardware to verify runtime correctness.

#include <cstdio>

__global__ void report_arch() {
#ifdef __CUDA_ARCH__
    printf("__CUDA_ARCH__ = %d\n", __CUDA_ARCH__);
#else
    printf("__CUDA_ARCH__ not defined\n");
#endif

#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
    printf("__CUDA_ARCH_FAMILY_SPECIFIC__ = %d\n", __CUDA_ARCH_FAMILY_SPECIFIC__);
#else
    printf("__CUDA_ARCH_FAMILY_SPECIFIC__ not defined\n");
#endif
}

// Use NVIDIA's own cuda_fp4.hpp conversion function to test E2M1
// This uses __CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__
// which gates the cvt.rn.satfinite.e2m1x2.f32 instruction
#include <cuda_fp4.h>

__global__ void test_e2m1_conversion() {
    // Convert two known float values to FP4 E2M1
    float val1 = 1.5f;
    float val2 = 0.5f;
    float2 input = make_float2(val1, val2);

    __nv_fp4x2_storage_t result = __nv_cvt_float2_to_fp4x2(
        input, __NV_E2M1, cudaRoundNearest);

    printf("E2M1 conversion: float2(%f, %f) -> 0x%02x\n",
           val1, val2, (unsigned)result);

    // Verify: E2M1 format for 1.5 should be 0b111 (exp=11, man=1)
    //         E2M1 format for 0.5 should be 0b010 (exp=01, man=0)
    // Packed: low nibble = val2, high nibble = val1
    unsigned expected_val1 = 7;  // 0b111
    unsigned expected_val2 = 2;  // 0b010
    unsigned expected = (expected_val1 << 4) | expected_val2;

    if ((unsigned)result == expected) {
        printf("PASS: Got expected 0x%02x\n", expected);
    } else {
        printf("FAIL: Expected 0x%02x, got 0x%02x\n", expected, (unsigned)result);
    }
}

int main() {
    printf("=== Architecture Info ===\n");
    report_arch<<<1,1>>>();
    cudaDeviceSynchronize();

    printf("\n=== E2M1 Conversion Test ===\n");
    test_e2m1_conversion<<<1,1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\n=== Device Info ===\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    return 0;
}
