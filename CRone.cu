#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
// #include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define M 128           // Number of BS antennas
#define N 32            // Number of users
#define SC  1          // Number of subcarriers
#define SYM 1           // Number of symbols
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

__global__ void matVec(cuComplex* A, cuComplex* r, cuComplex* m) 
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < SZ) {

		cuComplex tmp = {0.0f, 0.0f};

		for (int i = 0; i < N; i++) {
            tmp = cuCaddf(tmp, cuCmulf(r[idx*N + i], A[idx*N*N + i]));
		}
		m[idx*N] = tmp;
	}
}

__global__ void Caxpy(cuComplex alpha, cuComplex *p, cuComplex *r) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SZ)
    {
        p[idx*N] = cuCaddf(r[idx*N], cuCmulf(alpha, p[idx*N]));
    }
}

__global__ void Cymax(cuComplex beta, cuComplex *p, cuComplex *r) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SZ)
    {
        p[idx*N] = cuCsubf(r[idx*N], cuCmulf(beta, p[idx*N]));
    }
}


__global__ void Dot(cuComplex *r, cuComplex *m)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    cuComplex tmp = {0.0f, 0.0f};

    for (int i = 0; i < N; i++)
        tmp = cuCaddf(tmp, cuCmulf(cuConjf(r[idx*N + i]), m[idx*N + i]));
}

__global__ void Norm(cuComplex *e)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float tmpf = 0.0f;
    for (int i = 0; i < N; i++)
        tmpf += (e[idx*N + i].x * e[idx*N + i].x) + (e[idx*N + i].y * e[idx*N + i].y);
    tmpf = sqrtf(tmpf);
}

__global__ void CR (cuComplex *A, cuComplex *r, cuComplex *p, cuComplex *m, cuComplex *e, cuComplex *x)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < SZ) {

        float tmpf = 0.0f;
		cuComplex tmp = {0.0f, 0.0f};
        cuComplex alpha = {0.0f, 0.0f};
        cuComplex beta = {0.0f, 0.0f};
        cuComplex alpha_beta = {0.0f, 0.0f};

		for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++){
                tmp = cuCaddf(tmp, cuCmulf(r[idx*N + i], A[idx*N*N + i*N + j]));
            }

            m[idx*N + i] = tmp;
            e[idx*N + i] = tmp;
            // alpha divident
            alpha = cuCaddf(alpha, cuCmulf(cuConjf(r[idx*N + i]), tmp));
            tmpf += (tmp.x * tmp.x) + (tmp.y * tmp.y);
		}
        // alpha divisor
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;

        // alpha
        alpha.x = alpha.x / tmpf;
        alpha.y = alpha.y / tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCmulf(alpha, p[idx*N + i]);
            r[idx*N + i] = cuCsubf(r[idx*N + i], cuCmulf(alpha, e[idx*N + i]));
        }

        beta = {0.0f, 0.0f};
		for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++)
                tmp = cuCaddf(tmp, cuCmulf(r[idx*N + i], A[idx*N*N + i*N + j]));

            m[idx*N + i] = tmp;
            // beta divident
            beta = cuCaddf(beta, cuCmulf(cuConjf(r[idx*N + i]), tmp));
		}

        alpha = beta;
        beta = cuCdivf(beta, alpha_beta);

        tmpf = 0.0f;
        for (int i = 0; i < N; i++)
        {
            p[idx*N + i] = cuCaddf(r[idx*N + i], cuCmulf(beta, p[idx*N + i]));
            e[idx*N + i] = cuCaddf(m[idx*N + i], cuCmulf(beta, e[idx*N + i]));
            tmpf += (e[idx*N + i].x * e[idx*N + i].x) + (e[idx*N + i].y * e[idx*N + i].y);
        }
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;
        alpha.x = alpha.x/tmpf;
        alpha.y = alpha.y/tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCaddf(x[idx*N + i], cuCmulf(alpha, p[idx*N + i]));
            r[idx*N + i] = cuCsubf(r[idx*N + i], cuCmulf(alpha, e[idx*N + i]));
        }

        beta = {0.0f, 0.0f};
        for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++)
                tmp = cuCaddf(tmp, cuCmulf(r[idx*N + i], A[idx*N*N + i*N + j]));

            m[idx*N + i] = tmp;
            beta = cuCaddf(beta, cuCmulf(cuConjf(r[idx*N + i]), tmp));
		}
        alpha = beta;
        beta = cuCdivf(beta, alpha_beta);

        tmpf = 0.0f;
        for (int i = 0; i < N; i++)
        {
            p[idx*N + i] = cuCaddf(r[idx*N + i], cuCmulf(beta, p[idx*N + i]));
            e[idx*N + i] = cuCaddf(m[idx*N + i], cuCmulf(beta, e[idx*N + i]));
            tmpf += (e[idx*N + i].x * e[idx*N + i].x) + (e[idx*N + i].y * e[idx*N + i].y);
        }
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;
        alpha.x = alpha.x/tmpf;
        alpha.y = alpha.y/tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCaddf(x[idx*N + i], cuCmulf(alpha, p[idx*N + i]));
        }
	}
}

__global__ void CRVec (cuComplex *A, cuComplex *r, cuComplex *p, cuComplex *m, cuComplex *e, cuComplex *x)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ cuComplex r_s[SC*N];
    if(idx < SZ)
    {
        for (int i = 0; i < N; i++)
            r_s[threadIdx.x + i] = r[idx*N + i];
    }

	if (idx < SZ) {

        float tmpf = 0.0f;
		cuComplex tmp = {0.0f, 0.0f};
        cuComplex alpha = {0.0f, 0.0f};
        cuComplex beta = {0.0f, 0.0f};
        cuComplex alpha_beta = {0.0f, 0.0f};

		for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++){
                tmp = cuCaddf(tmp, cuCmulf(r_s[threadIdx.x + i], A[idx*N*N + i*N + j]));
                // printf("Tid %d R %f A%f\n", idx, r[idx*N + i].x, A[idx*N*N + i*N + j].y);
            }

            m[idx*N + i] = tmp;
            e[idx*N + i] = tmp;
            // alpha divident
            alpha = cuCaddf(alpha, cuCmulf(cuConjf(r_s[threadIdx.x + i]), tmp));
            tmpf += (tmp.x * tmp.x) + (tmp.y * tmp.y);
		}
        // alpha divisor
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;

        // alpha
        alpha.x = alpha.x / tmpf;
        alpha.y = alpha.y / tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCmulf(alpha, p[idx*N + i]);
            r_s[threadIdx.x + i] = cuCsubf(r_s[threadIdx.x + i], cuCmulf(alpha, e[idx*N + i]));
        }

        beta = {0.0f, 0.0f};
		for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++)
                tmp = cuCaddf(tmp, cuCmulf(r_s[threadIdx.x + i], A[idx*N*N + i*N + j]));

            m[idx*N + i] = tmp;
            // beta divident
            beta = cuCaddf(beta, cuCmulf(cuConjf(r_s[threadIdx.x + i]), tmp));
		}

        alpha = beta;
        beta = cuCdivf(beta, alpha_beta);

        tmpf = 0.0f;
        for (int i = 0; i < N; i++)
        {
            p[idx*N + i] = cuCaddf(r_s[threadIdx.x + i], cuCmulf(beta, p[idx*N + i]));
            e[idx*N + i] = cuCaddf(m[idx*N + i], cuCmulf(beta, e[idx*N + i]));
            tmpf += (e[idx*N + i].x * e[idx*N + i].x) + (e[idx*N + i].y * e[idx*N + i].y);
        }
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;
        alpha.x = alpha.x/tmpf;
        alpha.y = alpha.y/tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCaddf(x[idx*N + i], cuCmulf(alpha, p[idx*N + i]));
            r_s[threadIdx.x + i] = cuCsubf(r_s[threadIdx.x + i], cuCmulf(alpha, e[idx*N + i]));
        }

        beta = {0.0f, 0.0f};
        for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++)
                tmp = cuCaddf(tmp, cuCmulf(r_s[threadIdx.x + i], A[idx*N*N + i*N + j]));

            m[idx*N + i] = tmp;
            beta = cuCaddf(beta, cuCmulf(cuConjf(r_s[threadIdx.x + i]), tmp));
		}
        alpha = beta;
        beta = cuCdivf(beta, alpha_beta);

        tmpf = 0.0f;
        for (int i = 0; i < N; i++)
        {
            p[idx*N + i] = cuCaddf(r_s[threadIdx.x + i], cuCmulf(beta, p[idx*N + i]));
            e[idx*N + i] = cuCaddf(m[idx*N + i], cuCmulf(beta, e[idx*N + i]));
            tmpf += (e[idx*N + i].x * e[idx*N + i].x) + (e[idx*N + i].y * e[idx*N + i].y);
        }
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;
        alpha.x = alpha.x/tmpf;
        alpha.y = alpha.y/tmpf;

        for (int i = 0; i < N; i++)
            x[idx*N + i] = cuCaddf(x[idx*N + i], cuCmulf(alpha, p[idx*N + i]));
	}
}

__global__ void CRMat (cuComplex *A, cuComplex *r, cuComplex *p, cuComplex *m, cuComplex *e, cuComplex *x)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ cuComplex G[SC*N*N];

    if(idx < SZ)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++){
                G[threadIdx.x*N + i*N + j] = A[idx*N*N + i*N + j];
            }
    }

	if (idx < SZ) 
    {

        float tmpf = 0.0f;
		cuComplex tmp = {0.0f, 0.0f};
        cuComplex alpha = {0.0f, 0.0f};
        cuComplex beta = {0.0f, 0.0f};
        cuComplex alpha_beta = {0.0f, 0.0f};

		for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++){
                tmp = cuCaddf(tmp, cuCmulf(r[idx*N + i], G[threadIdx.x + i*N + j]));
            }

            m[idx*N + i] = tmp;
            e[idx*N + i] = tmp;
            // alpha divident
            alpha = cuCaddf(alpha, cuCmulf(cuConjf(r[idx*N + i]), tmp));
            tmpf += (tmp.x * tmp.x) + (tmp.y * tmp.y);
		}
        // alpha divisor
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;

        // alpha
        alpha.x = alpha.x / tmpf;
        alpha.y = alpha.y / tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCmulf(alpha, p[idx*N + i]);
            r[idx*N + i] = cuCsubf(r[idx*N + i], cuCmulf(alpha, e[idx*N + i]));
        }

        beta = {0.0f, 0.0f};
		for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++)
                tmp = cuCaddf(tmp, cuCmulf(r[idx*N + i], G[threadIdx.x + i*N + j]));

            m[idx*N + i] = tmp;
            // beta divident
            beta = cuCaddf(beta, cuCmulf(cuConjf(r[idx*N + i]), tmp));
		}

        alpha = beta;
        beta = cuCdivf(beta, alpha_beta);

        tmpf = 0.0f;
        for (int i = 0; i < N; i++)
        {
            p[idx*N + i] = cuCaddf(r[idx*N + i], cuCmulf(beta, p[idx*N + i]));
            e[idx*N + i] = cuCaddf(m[idx*N + i], cuCmulf(beta, e[idx*N + i]));
            tmpf += (e[idx*N + i].x * e[idx*N + i].x) + (e[idx*N + i].y * e[idx*N + i].y);
        }
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;
        alpha.x = alpha.x/tmpf;
        alpha.y = alpha.y/tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCaddf(x[idx*N + i], cuCmulf(alpha, p[idx*N + i]));
            r[idx*N + i] = cuCsubf(r[idx*N + i], cuCmulf(alpha, e[idx*N + i]));
        }

        beta = {0.0f, 0.0f};
        for (int i = 0; i < N; i++) {
            tmp = {0.0f, 0.0f};
            for (int j = 0; j < N; j++)
                tmp = cuCaddf(tmp, cuCmulf(r[idx*N + i], G[threadIdx.x + i*N + j]));

            m[idx*N + i] = tmp;
            beta = cuCaddf(beta, cuCmulf(cuConjf(r[idx*N + i]), tmp));
		}
        alpha = beta;
        beta = cuCdivf(beta, alpha_beta);

        tmpf = 0.0f;
        for (int i = 0; i < N; i++)
        {
            p[idx*N + i] = cuCaddf(r[idx*N + i], cuCmulf(beta, p[idx*N + i]));
            e[idx*N + i] = cuCaddf(m[idx*N + i], cuCmulf(beta, e[idx*N + i]));
            tmpf += (e[idx*N + i].x * e[idx*N + i].x) + (e[idx*N + i].y * e[idx*N + i].y);
        }
        tmpf = sqrtf(tmpf);
        alpha_beta = alpha;
        alpha.x = alpha.x/tmpf;
        alpha.y = alpha.y/tmpf;

        for (int i = 0; i < N; i++)
        {
            x[idx*N + i] = cuCaddf(x[idx*N + i], cuCmulf(alpha, p[idx*N + i]));
        }
	}
}

int main (void)
{
    // Get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    // Set up device
	CHECK (cudaSetDevice(dev));

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

    dim3 block (SC);
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

    /*** PREPROCESSING ***/
    /** Compute Gramian G = H^H*H **/
    createA(A);
    CHECK (cudaMemcpy(G, A, N*SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));

    // /** Regularize A --> G + s^2*I **/

    /** Compute ymf = HH*y **/
    createYMF(yMF_H);
    CHECK (cudaMemcpy(r, yMF_H, SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK (cudaMemcpy(p, yMF_H, SZ*sizeof(cuComplex), cudaMemcpyHostToDevice));

    /***---CONJUGATE RESIDUAL METHOD---***/
    CR <<< grid, block >>> (G, r, p, m, e, x);
    CHECK (cudaMemcpy(x_H, x, SZ*sizeof(cuComplex), cudaMemcpyDeviceToHost));

    printf ("\nDetected Vector: \n");
    for (int j = 0; j < N; j++) {
        printf ("%.2f %.1f ", x_H[SZ - j-1].x, x_H[SZ - j-1].y);
    printf ("\n");
    }

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
    free(H);
    free(HH);
    free(y);
    free(yMF_H);
    free(A);

    return EXIT_SUCCESS;
}