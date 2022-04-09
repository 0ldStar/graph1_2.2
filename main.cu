#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void subKernel(double *range, double first, unsigned curSize, unsigned size, int k/*, double *firstElArr*/) {
    unsigned thrX = threadIdx.x;
    unsigned blX = blockIdx.x;
    unsigned ind = blX * size + thrX;
    unsigned i = (ind / curSize * size + size - 1 - ind % curSize + k * size); // reverse
    double factor = range[(i / size) * size + k - 1] / first;
    if (i < size * size) {
        range[i] -= range[(k - 1) * size + i % size] * factor;
    }
}

__device__ void diagonalMultiplication(double *rez, const double *matrix, const unsigned *SIZE) {
    *rez = 1;
    for (unsigned i = *SIZE; i >= 1; --i) *rez *= matrix[(*SIZE + 1) * (*SIZE - i)];
}

int zeroesCheck(const double *range, const unsigned n, const unsigned *SIZE) {
    int count = 0, flag = 1;
    for (unsigned i = n; i < *SIZE; ++i)
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

void swap(double *range, unsigned n, const unsigned *SIZE) {
    double tmp;
    for (unsigned i = n; i < n + *SIZE; ++i) {
        tmp = range[i];
        range[i] = range[i + *SIZE];
        range[i + *SIZE] = tmp;
    }
}

int sort(double *matrix, const unsigned *SIZE) {
    int count = 0;
    for (unsigned i = 0; i < (*SIZE - 1) * *SIZE; i += *SIZE)
        if (zeroesCheck(matrix, i, SIZE) > zeroesCheck(matrix, i + *SIZE, SIZE)) {
            count++;
            swap(matrix, i, SIZE);
        }
    return power(-1, count);
}

__global__ void gaussianDeterminant(double *rez, double *matrix, unsigned *SIZE) {
    unsigned size = *SIZE;
    int flag = 1;
    int k = 1;
    while (size > 1) {
        unsigned threadX;
        unsigned blockX;
        if (size * size - size > 1024) {
            threadX = 1024;
            blockX = 1 + (size * size - size) / 1024;
        } else {
            threadX = size * size - size;
            blockX = 1;
        }
        dim3 threads = {threadX, 1, 1};
        dim3 blocks = {blockX, 1, 1};
        if (matrix[(*SIZE + 1) * (*SIZE - size)] == 0) {
            flag = 0;
            break;
        }
        double first = matrix[(*SIZE + 1) * (k - 1)];
        subKernel<<<blocks, threads>>>(matrix, first, size, *SIZE, k);
        cudaDeviceSynchronize();
        size--;
        k++;
    }
    if (flag)
        diagonalMultiplication(rez, matrix, SIZE);
    else *rez = 0;
}

void printMatrix(double *matrix, const unsigned *SIZE) {
    for (int i = 0; i < *SIZE * *SIZE; ++i) {
        if (i % *SIZE == 0)printf("\n");
        printf("%f ", matrix[i]);
    }
    printf("\n");
}

void zeroes(double *matrix, const unsigned *SIZE) {
    for (int i = 0; i < *SIZE * *SIZE; ++i) {
        matrix[i] = 0;
    }
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
    double *dMatrix, *dRez;
    cudaMalloc((void **) &dRez, sizeof(double));
    unsigned *dSIZE;
    cudaMalloc((void **) &dSIZE, sizeof(unsigned));
    unsigned SIZE, sign;
    cudaError_t cudaStatus;
    clock_t time_start, time_finish;
    while (fscanf(fp1, "%d", &SIZE) == 1) {
        time_start = clock();
        cudaMalloc((void **) &dMatrix, SIZE * SIZE * sizeof(double));
        matrix = (double *) malloc(SIZE * SIZE * sizeof(double));
        if (!matrix)exit(-3);
        for (int i = 0; i < SIZE * SIZE; ++i) {
            fscanf(fp1, "%lf", &matrix[i]);
        }
        sign = sort(matrix, &SIZE);
        cudaStatus = cudaMemcpy(dMatrix, matrix, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed1: \n");
            exit(-3);
        }
//        printf("Memcpy1 %ld \n", clock() - time_start);
        cudaStatus = cudaMemcpy(dSIZE, &SIZE, sizeof(unsigned), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed2: \n");
            exit(-4);
        }
//        printf("Memcpy2 %ld \n", clock() - time_start);
        gaussianDeterminant<<<1, 1>>>(dRez, dMatrix, dSIZE);
        cudaDeviceSynchronize();
//        printf("Determinant %ld \n", clock() - time_start);
        cudaStatus = cudaMemcpy(&determinant, dRez, sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed3\n");
            exit(-5);
        }
        zeroes(matrix, &SIZE);
        cudaStatus = cudaMemcpy(matrix, dMatrix, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed3\n");
            exit(-5);
        }
//        printf("Memcpy3 %ld \n", clock() - time_start);
        determinant *= (double) sign;
        time_finish = clock();
        fprintf(fp2, "%ld %f\n", time_finish - time_start, determinant);
        printf("\t %d\n", SIZE);
//        if (SIZE == 5) break;
//        if (determinant > DBL_MAX) {
//            perror("Determinant over\n");
//            exit(-2); }
        free(matrix);
        cudaFree(dMatrix);
    }    //printf("%f\n", *rez);
    cudaFree(dRez);
    cudaFree(dSIZE);
    fclose(fp1);
    fclose(fp2);
}

int main() {
    init();
    return 0;
}