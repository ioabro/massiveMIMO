#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define M 128           // Number of BS antennas
#define N 32            // Number of users
#define SC  1           // Number of subcarriers
#define SYM 1          // Number of symbols
#define NCH SC*SYM      // Number of channel matrices
#define SZ  SC*SYM*N    // Total threads

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    } \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

void createA(cuComplex *A)
{
    for (int i = 0; i < N*SZ; i++)
    {
        A[i].x = 0.123f;    
        A[i].y = 0.123f;     
    }
}

void createYMF(cuComplex *y)
{
    for (int i = 0; i < SZ; i++)
    {
        y[i].x = 0.123f;    
        y[i].y = 0.456f;     
    }
}

__global__ void ScaleA (cuComplex *A, float a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int d = idx + threadIdx.x;

    if (idx < SZ)
    {
        A[d].x = a * A[d].x;
        A[d].y = a * A[d].y;
    }
}

int main (void)
{
    cublasHandle_t handle;

    // Initialize cuBLAS library
    CHECK_CUBLAS (cublasCreate(&handle));
    
    // Get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    // Set up device
	CHECK (cudaSetDevice(dev));

    // al, bet - scalars for cuBLAS
    cuComplex  al = {1.0f, 0.0f};
    cuComplex  bet = {1.0f, 0.0f};

    cuComplex * H;          // Chanel matrix in host
    cuComplex * HH;         // Hermitian Chanel matrix in host

    cuComplex * H_dev;      // Chanel matrix in device
    cuComplex * HH_dev;     // Hermitian Chanel matrix in device

    cuComplex * G;          // Gram matrix in device

    cuComplex * y;          // Received vector in host
    cuComplex * y_dev;      // Received vector in device

    cuComplex * yMF_H;      // Matched filter in host
    cuComplex * yMF;        // Matched filter in device

    cuComplex * A;          // Matrix A in host

    cuComplex * x_H;        // Detected vector 

    // Five vectors of storage are needed in addition to the matrix A: x, p, Ap, r, Ar
    cuComplex * x;
    cuComplex * p;
    cuComplex * e; // Ap
    cuComplex * r;
    cuComplex * m; // Ar

    // al, bet - scalars
    float alpha_divisor;

    cuComplex alpha = {0.0f, 0.0f};
    cuComplex alpha_divident;
    cuComplex beta = {0.0f, 0.0f};
    cuComplex beta_divident;
    cuComplex beta_divisor;

    // Matrix H in host
    H = (cuComplex*) malloc (NCH * M * N * sizeof(cuComplex));
    if (!H) {
        printf ("Host memory allocation failed for H");
        return EXIT_FAILURE;
    }

    // Matrix H in host
    HH = (cuComplex*)malloc (NCH * M * N * sizeof(cuComplex));
    if (!HH) {
        printf ("Host memory allocation failed for HH");
        return EXIT_FAILURE;
    }

    // Received vector y in host
    y = (cuComplex*)malloc (NCH * M * sizeof(cuComplex));
    if (!y) {
        printf ("Host memory allocation failed for y");
        return EXIT_FAILURE;
    }

    // Detected vector y in host
    yMF_H = (cuComplex*)malloc (NCH * N * sizeof(cuComplex));
    if (!yMF_H) {
        printf ("Host memory allocation failed for yMF_H");
        return EXIT_FAILURE;
    }

    // Detected vector y in host
    x_H = (cuComplex*)malloc (NCH * N * sizeof(cuComplex));
    if (!x_H) {
        printf ("Host memory allocation failed for x_H");
        return EXIT_FAILURE;
    }

    // Matrix H in host
    A = (cuComplex*)malloc (NCH * N * N * sizeof(cuComplex));
    if (!A) {
        printf ("Host memory allocation failed for A");
        return EXIT_FAILURE;
    }

    // Allocate matrix H in device
    CHECK (cudaMalloc ((void**)&H_dev, NCH*M*N*sizeof(cuComplex)));

    // Allocate matrix HH in device
    CHECK (cudaMalloc ((void**)&HH_dev, NCH*M*N*sizeof(cuComplex)));

    // Allocate vector y in device
    CHECK (cudaMalloc ((void**)&y_dev, NCH*M*sizeof(cuComplex)));

    // Allocate vector yMF in device
    CHECK (cudaMalloc ((void**)&yMF, NCH*N*sizeof(cuComplex)));

    // Allocate matrix G in device
    CHECK (cudaMalloc ((void**)&G, NCH*N*N*sizeof(cuComplex)));

    // Allocate vector x in device
    CHECK (cudaMalloc ((void**)&x, NCH*N*sizeof(cuComplex)));

    // Allocate vector p in device
    CHECK (cudaMalloc ((void**)&p, NCH*N*sizeof(cuComplex)));

    // Allocate vector Ap in device
    CHECK (cudaMalloc ((void**)&e, N*sizeof(cuComplex)));

    // Allocate vector r in device
    CHECK (cudaMalloc ((void**)&r, NCH*N*sizeof(cuComplex)));
    
    // Allocate vector Ar in device
    CHECK (cudaMalloc ((void**)&m, N*sizeof(cuComplex)));

    // CHECK_CUBLAS (cublasSetMatrix (M, N, sizeof(*H), H, M, H_dev, M));

    // CHECK_CUBLAS (cublasSetMatrix (N, M, sizeof(*H), HH, N, HH_dev, N));

    // cublasSetVector() copies a vector x on the host to a vector on the GPU
    // CHECK_CUBLAS (cublasSetVector(M, sizeof (*y), y, 1, y_dev, 1)); //cp y->d_y

    /*** PREPROCESSING ***/
    /** Compute Gramian G = H^H*H **/
    // CHECK_CUBLAS (cublasCgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, M,
    //                         &al, HH_dev, N, H_dev, M,
    //                         &bet, G, N));
    createA(A);
    CHECK (cudaMemcpy(G, A, N*SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));

    // /** Regularize A --> G + s^2*I **/
    // float s2 = 2.0f;
    // ScaleA <<< grid, block >>> (G, s2);
    // cudaDeviceSynchronize();

    /** Compute ymf = HH*y **/
    // CHECK_CUBLAS (cublasCgemv (handle, CUBLAS_OP_N, N, M, &al,
    //                             HH_dev, N, y_dev, 1, &bet, yMF, 1));
    createYMF(yMF_H);
    CHECK (cudaMemcpy(yMF, yMF_H, SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));

    // Copy yMF -> p
    CHECK_CUBLAS (cublasCcopy (handle, N, yMF, 1, p, 1));

    // Copy yMF -> r
    CHECK_CUBLAS (cublasCcopy (handle, N, yMF, 1, r, 1));

    /***---CONJUGATE RESIDUAL METHOD---***/

    /** Compute Ap0 = Ar0 = e0 = m0 **/
    CHECK_CUBLAS (cublasCgemv (handle, CUBLAS_OP_N, N, N,&al,
                                G, N, yMF, 1, &bet, m, 1));
    // Copy m -> e
    CHECK_CUBLAS (cublasCcopy (handle, N, m, 1, e, 1));

    for (int i = 0; i < 3; i++)
    {
        // Compute r^H dot m
        CHECK_CUBLAS (cublasCdotc (handle, N, r, 1, m, 1, &alpha_divident));

        // Compute ||e||2
        CHECK_CUBLAS (cublasScnrm2(handle, N, e, 1, &alpha_divisor));


        /** Calculate a **/
        alpha.x = alpha_divident.x / alpha_divisor;
        alpha.y = alpha_divident.y / alpha_divisor;

        // /** Compute x **/
        CHECK_CUBLAS ( cublasCaxpy (handle, N, &alpha, p, 1, x, 1));
        
        // /** Compute r **/
        alpha.x = -alpha.x;
        alpha.y = -alpha.y;
        CHECK_CUBLAS (cublasCaxpy (handle, N, &alpha, e, 1, r, 1));

        /** Compute Ar **/
        CHECK_CUBLAS (cublasCgemv (handle, CUBLAS_OP_N, N, N, &al,
                                    G, N, r, 1, &bet, m, 1));

        // Compute r^H dot m
        CHECK_CUBLAS (cublasCdotc (handle, N, r, 1, m, 1, &beta_divident));

        /** Calculate b **/
        beta_divisor = alpha_divident;
        beta = cuCdivf(beta_divident, beta_divisor);

        // Compute p
        CHECK_CUBLAS (cublasCaxpy (handle, N, &beta, p, 1, r, 1));

        // // Compute e
        CHECK_CUBLAS (cublasCaxpy (handle, N, &alpha, e, 1, m, 1));
    }

    CHECK_CUBLAS (cublasGetMatrix (N, N, sizeof(*A), G, N, A, N));

    CHECK_CUBLAS (cublasGetVector (N, sizeof(*x), x, 1, x_H, 1));

    printf ("\nDetected Vector: \n");
    for (int j = 0; j < N; j++) {
        printf ("%.2f %.1f ", x_H[SZ - j - 1].x, x_H[SZ - j - 1].y);
    printf ("\n");
    }

    // destroy CUBLAS context
    cublasDestroy(handle);

    // free device memory
    cudaFree(H_dev);
    cudaFree(G); 
    cudaFree(yMF); 
    cudaFree(x); 
    cudaFree(p); 
    cudaFree(r); 
    cudaFree(e);
    cudaFree(m);

    // free host memory
    free(x_H);
    free(H);
    free(HH);
    free(y);
    free(yMF_H);
    free(A);

    return EXIT_SUCCESS;
}

// nvcc -arch sm_50 CR.cu -lcublas -o a
// nvprof ./a
// nvprof --print-gpu-trace a