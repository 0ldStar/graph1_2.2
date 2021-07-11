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

double diagonalMultiplication(const double *matrix, const int *SIZE) {
    double rez = 1;
    for (int i = *SIZE; i >= 1; --i) rez *= matrix[(*SIZE + 1) * (*SIZE - i)];
    return rez;
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

double gaussianDeterminant(double *matrix, int *SIZE) {
    int size = *SIZE;
    double first, factor;
    double *dMatrix;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **) &dMatrix, *SIZE * *SIZE * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(-3);
    }
    cudaStatus = cudaMemcpy(dMatrix, matrix, *SIZE * *SIZE * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 failed!");
        exit(-3);
    }
    while (size > 1) {
        if (matrix[(*SIZE + 1) * (*SIZE - size)] == 0) return 0;
        first = matrix[(*SIZE + 1) * (*SIZE - size)];
        for (int i = (*SIZE + 1) * (*SIZE - size) + *SIZE; i < *SIZE * *SIZE; i += *SIZE) {
            factor = matrix[i] / first;
            subKernel<<<1, size>>>(dMatrix, factor, i, (*SIZE + 1) * (*SIZE - size));
            cudaStatus = cudaMemcpy(matrix, dMatrix, *SIZE * *SIZE * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy2 failed!");
                exit(-3);
            }
        }
        size--;
    }
    cudaFree(dMatrix);
    return diagonalMultiplication(matrix, SIZE);
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
    double determinant;
    int SIZE, sign;
    clock_t time_start, time_finish;
    while (fscanf(fp1, "%d", &SIZE) == 1) {
        matrix = (double *) malloc(SIZE * SIZE * sizeof(double));
        if (!matrix)exit(-3);
        for (int i = 0; i < SIZE * SIZE; ++i) {
            fscanf(fp1, "%lf", &matrix[i]);
        }
        time_start = clock();
        sign = sort(matrix, &SIZE);
        determinant = gaussianDeterminant(matrix, &SIZE) * (double) sign;
        time_finish = clock();
        fprintf(fp2, "%ld %f\n", time_finish - time_start, determinant);
        if (determinant > DBL_MAX) exit(-2);
        free(matrix);
    }
    fclose(fp1);
    fclose(fp2);
}

int main() {
    init();
    return 0;
}
