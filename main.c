#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

void printMatrix(double *matrix, const int *SIZE) {
    for (int i = 0; i < *SIZE * *SIZE; ++i) {
        if (i % *SIZE == 0)printf("\n");
        printf("%f ", matrix[i]);
    }
    printf("\n");
}

double diagonalMultiplication(const double *matrix, const int *SIZE) {
    //printf("diag start\n");
    double rez = 1;
    for (int i = *SIZE; i >= 1; --i) rez *= matrix[(*SIZE + 1) * (*SIZE - i)];
    //printf("diag complete\n");
    return rez;
}

int zeroesCheck(const double *range, const int n, const int *SIZE) {

    //printf("check start\n");
    int count = 0, flag = 1;
    for (int i = n; i < *SIZE; ++i)
        if (range[i] == 0 && flag) {
            count++;
        } else flag = 0;

    //printf("check complete\n");
    return count;

}

int power(int a, int b) {
    int rez = 1;
    for (int i = 0; i < b; ++i) rez *= a;
    return rez;
}

void swap(double *range, int n, const int *SIZE) {

    //printf("swap start\n");
    double tmp;
    for (int i = n; i < n + *SIZE; ++i) {
        tmp = range[i];
        range[i] = range[i + *SIZE];
        range[i + *SIZE] = tmp;
    }

    //printf("swap complete\n");
}

int sort(double *matrix, const int *SIZE) {

    //printf("sort start\n");
    int count = 0;
    for (int i = 0; i < (*SIZE - 1) * *SIZE; i += *SIZE)
        if (zeroesCheck(matrix, i, SIZE) > zeroesCheck(matrix, i + *SIZE, SIZE)) {
            count++;
            swap(matrix, i, SIZE);
        }

    //printf("sort complete\n");
    return power(-1, count);
}

double gaussianDeterminant(double *matrix, int *SIZE) {

    //printf("det start\n");
    int size = *SIZE;
    double first, factor;
    while (size > 1) {
        //printf("s-%d\n", size);
        if (matrix[(*SIZE + 1) * (*SIZE - size)] == 0) return 0;
        first = matrix[(*SIZE + 1) * (*SIZE - size)];
        //printf("F-%f %d\n", first, (*SIZE + 1) * (*SIZE - size));
        //first = matrix[*SIZE - size][*SIZE - size];
        for (int i = (*SIZE + 1) * (*SIZE - size) + *SIZE; i < *SIZE * *SIZE; i += *SIZE) {
            //factor = matrix[i][*SIZE - size] / first;
            //printf("->%f %d\nS = %d\n", matrix[i], i, *SIZE);
            factor = matrix[i] / first;
            for (int j = 0; j < size; ++j) {
                //printf("%d %d -= %d %d\n", i, i + j, (*SIZE + 1) * (*SIZE - size), (*SIZE + 1) * (*SIZE - size) + j);
                matrix[i + j] -= matrix[(*SIZE + 1) * (*SIZE - size) + j] * factor;
                //matrix[i][j] -= matrix[*SIZE - size][j] * factor;
            }
        }
        size--;
    }
    //printMatrix(matrix, SIZE);
    //printf("det complete\n");
    return diagonalMultiplication(matrix, SIZE);
}

void init() {
    FILE *fp1, *fp2;
    if ((fp1 = fopen("read.txt", "r")) == NULL) {
        printf("Can't open file 'read.txt'\n");
        exit(-1);
    }
    if ((fp2 = fopen("write.txt", "w")) == NULL) {
        printf("Can't open file 'write.txt'\n");
        exit(-1);
    }
    double *matrix;
    double determinant;
    int SIZE, sign;
    clock_t time_start, time_finish;
    while (fscanf(fp1, "%d", &SIZE) == 1) {
        //printf("read start\n");
        matrix = (double *) malloc(SIZE * SIZE * sizeof(double));
        if (!matrix)exit(-3);
        for (int i = 0; i < SIZE * SIZE; ++i) {
            fscanf(fp1, "%lf", &matrix[i]);
        }
        //printMatrix(matrix, &SIZE);
        time_start = clock();
        sign = sort(matrix, &SIZE);
        determinant = gaussianDeterminant(matrix, &SIZE) * (double) sign;
        //determinant = SIZE;
        time_finish = clock();
        //printMatrix(matrix, &SIZE);
        fprintf(fp2, "%ld %f\n", time_finish - time_start, determinant);
        if (determinant > DBL_MAX) exit(-2);
        //printf("YEAP\n");
        free(matrix);
        //printf("HEAP\n");
        //printf("\t%f\n", determinant);
        //printf("read complete\n");
    }
    fclose(fp1);
    fclose(fp2);
}

int main() {
    init();
    return 0;
}
