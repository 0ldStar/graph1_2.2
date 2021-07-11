#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void subKernel(double *range, const double factor, int firstInd, int secondInd) {
    unsigned int i = threadIdx.x;
    range[firstInd + i] -= range[secondInd + i] * factor;
}

void printMatrix(double *matrix, const int *SIZE) {
    for (int i = 0; i < *SIZE * *SIZE; ++i) {
        if (i % *SIZE == 0)printf("\n");
        printf("%f ", matrix[i]);
    }
    printf("\n");
}

__device__ void diagonalMultiplication(double *rez, const double *matrix, const int *SIZE) {
    *rez = 1;
    for (int i = *SIZE; i >= 1; --i) *rez *= matrix[(*SIZE + 1) * (*SIZE - i)];
}

int zeroesCheck(const double *range, const int n, const int *SIZE) {
    int count = 0, flag = 1;
    for (int i = n; i < *SIZE; ++i)
        if (range[i] == 0 && flag) {
            count++;
        } else flag = 0;
    return count;
}

int power(const int a, int b) {
    int rez = 1;
    for (int i = 0; i < b; ++i) rez *= a;
    return rez;
}

void swap(double *range, int n, const int *SIZE) {
    double tmp;
    for (int i = n; i < n + *SIZE; ++i) {
        tmp = range[i];
        range[i] = range[i + *SIZE];
        range[i + *SIZE] = tmp;
    }
}

int sort(double *matrix, const int *SIZE) {
    int count = 0;
    for (int i = 0; i < (*SIZE - 1) * *SIZE; i += *SIZE)
        if (zeroesCheck(matrix, i, SIZE) > zeroesCheck(matrix, i + *SIZE, SIZE)) {
            count++;
            swap(matrix, i, SIZE);
        }
    return power(-1, count);
}

__global__ void gaussianDeterminant(double *rez, double *matrix, int *SIZE) {
    int size = *SIZE;
    double first, factor;
    __syncthreads();
    while (size > 1) {
        if (matrix[(*SIZE + 1) * (*SIZE - size)] == 0) goto exit;
        first = matrix[(*SIZE + 1) * (*SIZE - size)];
        for (int i = (*SIZE + 1) * (*SIZE - size) + *SIZE; i < *SIZE * *SIZE; i += *SIZE) {
            factor = matrix[i] / first;
            subKernel<<<1, size>>>(matrix, factor, i, (*SIZE + 1) * (*SIZE - size));
            cudaDeviceSynchronize();
        }
        size--;
    }
    diagonalMultiplication(rez, matrix, SIZE);
    //printf("%f\n", *rez);
    __syncthreads();
    exit:;
}

void init() {
    FILE *fp1, *fp2;
    if ((fp1 = fopen("read.txt", "r")) == nullptr) {
        printf("Can't open file 'read.txt'\n");
        exit(-1);
    }
    if ((fp2 = fopen("write.txt", "w")) == nullptr) {
        printf("Can't open file 'write.txt'\n");
        exit(-1);
    }
    double *matrix;
    double *dMatrix;
    double determinant;
    double *dRez;
    cudaMalloc((void **) &dRez, sizeof(double));
    int *dSIZE;
    cudaMalloc((void **) &dSIZE, sizeof(int));
    int SIZE, sign;
    cudaError_t cudaStatus;
    clock_t time_start, time_finish;
    while (fscanf(fp1, "%d", &SIZE) == 1) {
        cudaMalloc((void **) &dMatrix, SIZE * SIZE * sizeof(double));
        matrix = (double *) malloc(SIZE * SIZE * sizeof(double));
        if (!matrix)exit(-3);
        for (int i = 0; i < SIZE * SIZE; ++i) {
            fscanf(fp1, "%lf", &matrix[i]);
        }
        time_start = clock();
        sign = sort(matrix, &SIZE);
        cudaStatus = cudaMemcpy(dMatrix, matrix, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: \n");
            exit(-3);
        }
        cudaStatus = cudaMemcpy(dSIZE, &SIZE, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: \n");
            exit(-4);
        }
        gaussianDeterminant<<<1, 1>>>(dRez, dMatrix, dSIZE);
        cudaStatus = cudaMemcpy(&determinant, dRez, sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed\n");
            exit(-5);
        }
        determinant *= (double) sign;
        time_finish = clock();
        fprintf(fp2, "%ld %f\n", time_finish - time_start, determinant);
        if (determinant > DBL_MAX) exit(-2);
        free(matrix);
        cudaFree(dMatrix);
    }
    fclose(fp1);
    fclose(fp2);
}

int main() {
    init();
    return 0;
}
