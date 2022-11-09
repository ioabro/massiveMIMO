#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
// #include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define M 128           // Number of BS antennas
#define N 32            // Number of users
#define SC  29          // Number of subcarriers
#define SYM 284          // Number of symbols
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

// #define CHECK_CUBLAS(call)                                                     \
// {                                                                              \
//     cublasStatus_t err;                                                        \
//     if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
//     {                                                                          \
//         fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
//                 __LINE__);                                                     \
//         exit(1);                                                               \
//     }                                                                          \
// }

void createH(cuComplex *a, cuComplex *b)
{
    int i, j, k;

    for (k = 0; k < NCH; k++)
    {
        // define mxn matrix column by column                                  
        for (j = 0; j < N; j++){                            
            for (i = 0; i < M; i++){                                            
                a[k*M*N + IDX2C(i,j,M)].x = 0.123f;    
                a[k*M*N + IDX2C(i,j,M)].y = 0.789f;                                 
            }
        }

        // define nxm matrix column by column
        for (j = 0; j < M; j++){                            
            for (i = 0; i < N; i++){                                            
                b[k*M*N + IDX2C(i,j,N)].x = 0.123f;    
                b[k*M*N + IDX2C(i,j,N)].y = -0.789f;                                  
            }
        }
    }
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

void createReceived(cuComplex *R, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        R[i].x = (float)(rand() & 0xFF) / 10.0f;
        R[i].y = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

__global__ void ScaleA (cuComplex *A, float a)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = threadIdx.x * N + tid;

    if (idx < SZ)
    {
        A[idx].x = a * A[idx].x;
        A[idx].y = a * A[idx].y;
    }
}

// __global__ void CR (cuComplex* A, cuComplex* r, cuComplex* p, cuComplex* x) 
// {
// 	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     float alpha_divisor;

//     cuComplex alpha_beta = {0.0f, 0.0f};
//     cuComplex alpha_divident;
//     cuComplex beta_divident;

//     cuComplex m;
//     cuComplex e;

// 	if (idx < SZ) 
//     {
// 		m = {0.0f, 0.0f};
// 		for (int i = 0; i < N; i++) {
//             m = cuCaddf( m, cuCmulf(r[((idx>>5)<<5) + i], A[N * idx + i]) );
// 		}
        
//         e = m;
//         // __syncthreads();

//         // Dot
//         alpha_divident = cuCmulf(cuConjf(r[idx]), m);
        
//         // reduce & broadcast
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 16, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 8, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 4, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 2, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 1, 32);

//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 16, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 8, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 4, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 2, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 1, 32);
        
//         // Norm
//         alpha_divisor = (m.x * m.x) + (m.y * m.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);
        
//         // Sqrt
//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;
        

//         // AXPY
//         x[idx] = cuCmulf(alpha_beta, r[idx]); // First p same as r -- xo = 0
        
//         // AXPY
//         r[idx] = cuCsubf( r[idx], cuCmulf(alpha_beta, e) ); // still equal to m

//         // __syncthreads();

//         // Matvec
//         m = {0.0f, 0.0f};
// 		for (int i = 0; i < N; i++) {
//             m = cuCaddf( m, cuCmulf(r[((idx>>5)<<5) + i], A[N * idx + i]) );
// 		}

//         // __syncthreads();

//         // Dot
//         beta_divident = cuCmulf(cuConjf(r[idx]), m);

//         // reduce & broadcast
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 16, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 8, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 4, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 2, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 1, 32);

//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 16, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 8, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 4, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 2, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 1, 32);

//         // beta_divisor = alpha_divident;
//         alpha_beta = cuCdivf(beta_divident, alpha_divident);

//         // AXPY
//         p[idx] = cuCaddf( r[idx], cuCmulf(alpha_beta, p[idx]) );

//         // AXPY
//         e = cuCaddf( m,  cuCmulf(alpha_beta, e) );

//         //  Norm
//         alpha_divisor = (e.x * e.x) + (e.y * e.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_divident = beta_divident;
//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;
        
//         // AXPY
//         x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
//         // AXPY
//         r[idx] = cuCsubf( r[idx], cuCmulf(alpha_beta, e) );

//         // __syncthreads();

//         // Matvec
//         m = {0.0f, 0.0f};
// 		for (int i = 0; i < N; i++) {
//             m = cuCaddf( m, cuCmulf(r[((idx>>5)<<5) + i], A[N * idx + i]) );
//         }

//         beta_divident = cuCmulf(cuConjf(r[idx]), m);

//         // reduce & broadcast
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 16, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 8, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 4, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 2, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 1, 32);

//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 16, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 8, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 4, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 2, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 1, 32);

//         // beta_divisor = alpha_divident;
//         alpha_beta = cuCdivf(beta_divident, alpha_divident);
        
//         // AXPY
//         p[idx] = cuCaddf( r[idx], cuCmulf(alpha_beta, p[idx]) );

//         // AXPY
//         e = cuCaddf( m,  cuCmulf(alpha_beta, e) );

//         // Norm
//         alpha_divisor = (e.x * e.x) + (e.y * e.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_divident = beta_divident;
//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;

//         // AXPY
//         x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
// 	}
// }

// __global__ void CRMat (cuComplex* A, cuComplex* r, cuComplex* p, cuComplex* x) 
// {
// 	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     __shared__ cuComplex G[SC*N*N];

//     if(idx < SZ)
//     {
//         for (int i = 0; i < N; i++)
//             G[(((threadIdx.x>>5)<<5)<<5) + (threadIdx.x & 31) + (i<<5)] = A[(((idx>>5)<<5)<<5) + (threadIdx.x & 31) + (i<<5)];
//     }
//     // __syncthreads();
    
//     float alpha_divisor;

//     cuComplex alpha_beta = {0.0f, 0.0f};
//     cuComplex alpha_divident;
//     cuComplex beta_divident;

//     cuComplex m;
//     cuComplex e;

// 	if (idx < SZ) 
//     {
// 		m = {0.0f, 0.0f};
// 		for (int i = 0; i < N; i++) {
//             m = cuCaddf( m, cuCmulf(r[(idx/N)*N + i], G[threadIdx.x*N + i]) );
// 		}
        
//         e = m;
//         // __syncthreads();

//         // Dot
//         alpha_divident = cuCmulf(cuConjf(r[idx]), m);
        
//         // reduce & broadcast
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 16, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 8, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 4, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 2, 32);
//         alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 1, 32);

//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 16, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 8, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 4, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 2, 32);
//         alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 1, 32);
        
//         // Norm
//         alpha_divisor = (m.x * m.x) + (m.y * m.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);
//         // printf("Thread %d divi %f\n", idx, alpha_divisor);
//         // Sqrt
//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;
        

//         // AXPY
//         x[idx] = cuCmulf(alpha_beta, r[idx]); // First p same as r -- xo = 0
        
//         // AXPY
//         r[idx] = cuCsubf( r[idx], cuCmulf(alpha_beta, e) ); // still equal to m

//         // Matvec
//         m = {0.0f, 0.0f};
// 		for (int i = 0; i < N; i++) {
//             m = cuCaddf( m, cuCmulf(r[(idx/N)*N + i], G[threadIdx.x*N + i]) );
// 		}

//         // __syncthreads();

//         // Dot
//         beta_divident = cuCmulf(cuConjf(r[idx]), m);

//         // reduce & broadcast
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 16, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 8, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 4, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 2, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 1, 32);

//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 16, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 8, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 4, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 2, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 1, 32);

//         // beta_divisor = alpha_divident;
//         alpha_beta = cuCdivf(beta_divident, alpha_divident);

//         // AXPY
//         p[idx] = cuCaddf( r[idx], cuCmulf(alpha_beta, p[idx]) );

//         // AXPY
//         e = cuCaddf( m,  cuCmulf(alpha_beta, e) );

//         //  Norm
//         alpha_divisor = (e.x * e.x) + (e.y * e.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_divident = beta_divident;
//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;
        
//         // AXPY
//         x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
//         // AXPY
//         r[idx] = cuCsubf( r[idx], cuCmulf(alpha_beta, e) );

//         // __syncthreads();

//         // Matvec
//         m = {0.0f, 0.0f};
// 		for (int i = 0; i < N; i++) {
//             m = cuCaddf( m, cuCmulf(r[(idx/N)*N + i], G[threadIdx.x*N + i]) );
//         }

//         beta_divident = cuCmulf(cuConjf(r[idx]), m);

//         // __syncthreads();

//         // reduce & broadcast
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 16, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 8, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 4, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 2, 32);
//         beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 1, 32);

//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 16, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 8, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 4, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 2, 32);
//         beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 1, 32);

//         // beta_divisor = alpha_divident;
//         alpha_beta = cuCdivf(beta_divident, alpha_divident);
        
//         // AXPY
//         p[idx] = cuCaddf( r[idx], cuCmulf(alpha_beta, p[idx]) );

//         // AXPY
//         e = cuCaddf( m,  cuCmulf(alpha_beta, e) );

//         // Norm
//         alpha_divisor = (e.x * e.x) + (e.y * e.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_divident = beta_divident;
//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;

//         // AXPY
//         x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
// 	}
// }

__global__ void CRVec (cuComplex* A, cuComplex* r, cuComplex* p, cuComplex* x) 
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ cuComplex r_s[SC*N];

    if(idx < SZ)
        r_s[threadIdx.x] = r[idx];

    // __syncthreads();

    float alpha_divisor;

    cuComplex alpha_beta = {0.0f, 0.0f};
    cuComplex alpha_divident;
    cuComplex beta_divident;

    cuComplex m;
    cuComplex e;

	if (idx < SZ) 
    {
		m = {0.0f, 0.0f};
		for (int i = 0; i < N; i++) {
			m = cuCaddf( m, cuCmulf(r_s[((threadIdx.x>>5)<<5) + i], A[N * idx + i]) );
		}
        
        e = m;
        // __syncthreads();

        // Dot
        alpha_divident = cuCmulf(cuConjf(r_s[threadIdx.x]), m);
        
        // reduce & broadcast
        alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 16, 32);
        alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 8, 32);
        alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 4, 32);
        alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 2, 32);
        alpha_divident.x += __shfl_xor_sync(0xffffffff, alpha_divident.x, 1, 32);

        alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 16, 32);
        alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 8, 32);
        alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 4, 32);
        alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 2, 32);
        alpha_divident.y += __shfl_xor_sync(0xffffffff, alpha_divident.y, 1, 32);
        
        // Norm
        alpha_divisor = (m.x * m.x) + (m.y * m.y);
        // reduce & broadcast
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);
        // Sqrt
        alpha_divisor = sqrtf(alpha_divisor);

        alpha_beta.x = alpha_divident.x / alpha_divisor;
        alpha_beta.y = alpha_divident.y / alpha_divisor;
        

        // AXPY
        x[idx] = cuCmulf(alpha_beta, r_s[threadIdx.x]); // First p same as r -- xo = 0
        
        // AXPY
        r_s[threadIdx.x] = cuCsubf( r_s[threadIdx.x], cuCmulf(alpha_beta, e) ); // still equal to m

        // __syncthreads();

        // Matvec
        m = {0.0f, 0.0f};
		for (int i = 0; i < N; i++) {
            m = cuCaddf( m, cuCmulf(r_s[((threadIdx.x>>5)<<5) + i], A[N * idx + i]) );
		}

        // __syncthreads();

        // Dot
        beta_divident = cuCmulf(cuConjf(r_s[threadIdx.x]), m);

        // reduce & broadcast
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 16, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 8, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 4, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 2, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 1, 32);

        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 16, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 8, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 4, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 2, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 1, 32);

        // beta_divisor = alpha_divident;
        alpha_beta = cuCdivf(beta_divident, alpha_divident);

        // AXPY
        p[idx] = cuCaddf( r_s[threadIdx.x], cuCmulf(alpha_beta, p[idx]) );

        // AXPY
        e = cuCaddf( m,  cuCmulf(alpha_beta, e) );

        //  Norm
        alpha_divisor = (e.x * e.x) + (e.y * e.y);
        // reduce & broadcast
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

        alpha_divisor = sqrtf(alpha_divisor);

        alpha_divident = beta_divident;
        alpha_beta.x = alpha_divident.x / alpha_divisor;
        alpha_beta.y = alpha_divident.y / alpha_divisor;
        
        // AXPY
        x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
        // AXPY 
        r_s[threadIdx.x] = cuCsubf( r_s[threadIdx.x], cuCmulf(alpha_beta, e) ); 

        // __syncthreads();

        // Matvec
        m = {0.0f, 0.0f};
		for (int i = 0; i < N; i++) {
            m = cuCaddf( m, cuCmulf(r_s[((threadIdx.x>>5)<<5) + i], A[N * idx + i]) );
        }

        beta_divident = cuCmulf(cuConjf(r_s[threadIdx.x]), m);

        // __syncthreads();

        // reduce & broadcast
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 16, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 8, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 4, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 2, 32);
        beta_divident.x += __shfl_xor_sync(0xffffffff, beta_divident.x, 1, 32);

        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 16, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 8, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 4, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 2, 32);
        beta_divident.y += __shfl_xor_sync(0xffffffff, beta_divident.y, 1, 32);

        // beta_divisor = alpha_divident;
        alpha_beta = cuCdivf(beta_divident, alpha_divident);
        
        // AXPY
        p[idx] = cuCaddf( r_s[threadIdx.x], cuCmulf(alpha_beta, p[idx]) );

        // AXPY
        e = cuCaddf( m,  cuCmulf(alpha_beta, e) );

        // Norm
        alpha_divisor = (e.x * e.x) + (e.y * e.y);
        // reduce & broadcast
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
        alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

        alpha_divisor = sqrtf(alpha_divisor);

        alpha_divident = beta_divident;
        alpha_beta.x = alpha_divident.x / alpha_divisor;
        alpha_beta.y = alpha_divident.y / alpha_divisor;

        // AXPY
        x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
	}
}

// __global__ void CRShuf (cuComplex* A, cuComplex* r, cuComplex* p, cuComplex* x, cuComplex* m) 
// {
// 	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     float alpha_divisor;

//     cuComplex alpha_beta = {0.0f, 0.0f};
//     cuComplex alpha_divident = {0.0f, 0.0f};
//     cuComplex beta_divident = {0.0f, 0.0f};

//     cuComplex tmp;
//     cuComplex e;
//     cuComplex rr;

// 	if (idx < SZ) 
//     {
//         rr = r[idx];
// 		for (int i = 0; i < N; i++) {
//             tmp = cuCmulf(rr, A[(((idx>>5)<<5)<<5) + (threadIdx.x & 31) + (i<<5)]);

//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 16, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 8, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 4, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 2, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 1, 32);

//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 16, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 8, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 4, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 2, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 1, 32);
            
//             // Dot
//             alpha_divident = cuCaddf (alpha_divident, cuCmulf( cuConjf(rr), tmp) );

//             // Norm
//             alpha_divisor += (tmp.x * tmp.x) + (tmp.y * tmp.y);

//             if ((threadIdx.x & 31) == i)
//                 m[idx] = tmp;
//             __syncwarp();

//             // // printf("Thread %d Mod %d F %d tmp %.2f m %.2f divden %.2f divisor %.2f\n", threadIdx.x, \
//             //                      threadIdx.x & 31, \
//             //                      (threadIdx.x & 31) == i, \
//             //                     tmp.x, m[idx].x, alpha_divident.x, alpha_divisor); 
// 		}

//         // Sqrt
//         alpha_divisor = sqrtf(alpha_divisor);
        
//         e = m[idx];

//         // AXPY
//         x[idx] = cuCmulf(alpha_beta, rr); // First p same as r -- xo = 0
        
//         // AXPY
//         r[idx] = cuCsubf( rr, cuCmulf(alpha_beta, e) ); // still equal to m

//         // Matvec
//         rr = r[idx];
// 		for (int i = 0; i < N; i++) {
//             tmp = cuCmulf(rr, A[(((idx>>5)<<5)<<5) + (threadIdx.x & 31) + (i<<5)]);

//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 16, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 8, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 4, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 2, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 1, 32);

//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 16, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 8, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 4, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 2, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 1, 32);
            
//             // Dot
//             beta_divident = cuCaddf (beta_divident, cuCmulf( cuConjf(rr), tmp) );

//             if ((threadIdx.x & 31) == i)
//                 m[idx] = tmp;
//             __syncwarp();

// 		}

//         // beta_divisor = alpha_divident;
//         alpha_beta = cuCdivf(beta_divident, alpha_divident);

//         // AXPY
//         p[idx] = cuCaddf( rr, cuCmulf(alpha_beta, p[idx]) );

//         // AXPY
//         e = cuCaddf( m[idx],  cuCmulf(alpha_beta, e) );

//         //  Norm
//         alpha_divisor = (e.x * e.x) + (e.y * e.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_divident = beta_divident;
//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;
        
//         // AXPY
//         x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
//         // AXPY
//         r[idx] = cuCsubf( rr, cuCmulf(alpha_beta, e) );

//         // Matvec
//         rr = r[idx];
// 		for (int i = 0; i < N; i++) {
//             tmp = cuCmulf(rr, A[(((idx>>5)<<5)<<5) + (threadIdx.x & 31) + (i<<5)]);

//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 16, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 8, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 4, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 2, 32);
//             tmp.x += __shfl_xor_sync(0xffffffff, tmp.x, 1, 32);

//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 16, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 8, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 4, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 2, 32);
//             tmp.y += __shfl_xor_sync(0xffffffff, tmp.y, 1, 32);
            
//             // Dot
//             beta_divident = cuCaddf (beta_divident, cuCmulf( cuConjf(rr), tmp) );

//             if ((threadIdx.x & 31) == i)
//                 m[idx] = tmp;
//             __syncwarp();
// 		}

//         // beta_divisor = alpha_divident;
//         alpha_beta = cuCdivf(beta_divident, alpha_divident);
        
//         // AXPY
//         p[idx] = cuCaddf( rr, cuCmulf(alpha_beta, p[idx]) );

//         // AXPY
//         e = cuCaddf( m[idx],  cuCmulf(alpha_beta, e) );

//         // Norm
//         alpha_divisor = (e.x * e.x) + (e.y * e.y);
//         // reduce & broadcast
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 16, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 8, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 4, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 2, 32);
//         alpha_divisor += __shfl_xor_sync(0xffffffff, alpha_divisor, 1, 32);

//         alpha_divisor = sqrtf(alpha_divisor);

//         alpha_divident = beta_divident;
//         alpha_beta.x = alpha_divident.x / alpha_divisor;
//         alpha_beta.y = alpha_divident.y / alpha_divisor;

//         // AXPY
//         x[idx] = cuCaddf( x[idx], cuCmulf(alpha_beta, p[idx]) );
// 	}
// }

int main (void)
{
    int i, j;

    // cudaError_t cudaStat;
    // cublasStatus_t stat;
    // cublasHandle_t handle;

    // Initialize cuBLAS library
    // CHECK_CUBLAS (cublasCreate(&handle));
    
    // Get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    // Set up device
	CHECK (cudaSetDevice(dev));

    // al, bet - scalars for cuBLAS
    // cuComplex  al = {1.0f, 0.0f};
    // cuComplex  bet = {1.0f, 0.0f};

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
    // cuComplex * e;
    cuComplex * r;
    cuComplex * m;

    dim3 block (SC*32);
    dim3 grid  (SYM);

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

    // Create Channel
    createH(H, HH);

    // Received vector y in host
    y = (cuComplex*)malloc (NCH * M * sizeof(cuComplex));
    if (!y) {
        printf ("Host memory allocation failed for y");
        return EXIT_FAILURE;
    }

    // Create received vector
    // createReceived(y, M);
    for (i = 0; i < NCH * M; i++){
        y[i].x = 0.456f;
        y[i].y = 0.456f; 
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
    // CHECK (cudaMalloc ((void**)&e, N*sizeof(cuComplex)));

    // Allocate vector r in device
    CHECK (cudaMalloc ((void**)&r, NCH*N*sizeof(cuComplex)));
    
    // Allocate vector Ar in device
    CHECK (cudaMalloc ((void**)&m, NCH*N*sizeof(cuComplex)));

    // This function copies a tile of rows x cols elements from a matrix A
    // in host memory space to a matrix B in GPU memory space.
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
    // CHECK_CUBLAS (cublasSetMatrix (NCH*N, NCH*N, sizeof(cuComplex), A, NCH*N, G, NCH*N));
    CHECK (cudaMemcpy(G, A, N*SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));

    // /** Regularize A --> G + s^2*I **/
    // float s2 = 2.0f;
    // ScaleA <<< grid, block >>> (G, s2);
    // cudaDeviceSynchronize();

    /** Compute ymf = HH*y **/
    // This function performs the matrix-vector multiplication
    // CHECK_CUBLAS (cublasCgemv (handle, CUBLAS_OP_N, N, M, &al,
    //                             HH_dev, N, y_dev, 1, &bet, yMF, 1));
    createYMF(yMF_H);
    // CHECK_CUBLAS (cublasSetVector(NCH*N, sizeof (cuComplex), yMF_H, 1, yMF, 1));
    CHECK (cudaMemcpy(yMF, yMF_H, SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK (cudaMemcpy(p, yMF_H, SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));

    // CR <<< grid, block >>> (G, yMF, p, x);
    // CRMat <<< grid, block >>> (G, yMF, p, x);
    CRVec <<< grid, block >>> (G, yMF, p, x);
    // CRShuf <<< grid, block >>> (G, yMF, p, x, m);
    cudaDeviceSynchronize();
    
    // CHECK_CUBLAS (cublasGetVector (NCH*N, sizeof(cuComplex), x, 1, x_H, 1));
    CHECK (cudaMemcpy(x_H, x, SZ*sizeof(cuComplex), cudaMemcpyDeviceToHost));

    printf ("\nDetected Vector: \n");
    for (j = 0; j < N; j++) {
        printf ("%.2f %.1f ", x_H[SZ - j - 1].x, x_H[SZ - j - 1].y);
    printf ("\n");
    }

    // destroy CUBLAS context
    // cublasDestroy(handle);

    // free device memory
    cudaFree(H_dev);
    cudaFree(G); 
    cudaFree(yMF); 
    cudaFree(x); 
    cudaFree(p); 
    cudaFree(r); 
    // cudaFree(e);
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
