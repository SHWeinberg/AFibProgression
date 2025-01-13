#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>

// windows i/o
// #include <direct.h>

// linux i/o
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>


#ifndef cuda_afib_utils_h
#define cuda_afib_utils_h


void writeMatrixCsvFloatStateSave(float *mat, const char *write_file, int append_val, int width, int height){
    FILE *fp;
    if (append_val==0) fp = fopen(write_file, "w");
    else fp = fopen(write_file, "a");

    for (int ix = 0; ix < width; ix++)
    {
      for (int iy = 0; iy < height; iy++)
      {
          fprintf(fp, "%.9g,", mat[ix * width + iy]);
          if ((iy == height - 1) && (ix == width - 1))
              fprintf(fp, "\n");
      }
    }
    fclose(fp);
}

void writeMatrixCsvDoubleStateSave(double *mat, const char *write_file, int append_val, int width, int height){
    FILE *fp;
    if (append_val==0) fp = fopen(write_file, "w");
    else fp = fopen(write_file, "a");

    for (int ix = 0; ix < width; ix++)
    {
      for (int iy = 0; iy < height; iy++)
      {
          fprintf(fp, "%.17g,", mat[ix * width + iy]);
          if ((iy == height - 1) && (ix == width - 1))
              fprintf(fp, "\n");
      }
    }
    fclose(fp);
}

void writeMatrixCsvFloat(float *mat, const char *write_file, int append_val, int width, int height){
    FILE *fp;
    if (append_val==0) fp = fopen(write_file, "w");
    else fp = fopen(write_file, "a");

    for (int ix = 0; ix < width; ix++)
    {
      for (int iy = 0; iy < height; iy++)
      {
          fprintf(fp, "%.8f,", mat[ix * width + iy]);
          if ((iy == height - 1) && (ix == width - 1))
              fprintf(fp, "\n");
      }
    }
    fclose(fp);
}

void writeMatrixCsvDouble(double *mat, const char *write_file, int append_val, int width, int height){
    FILE *fp;
    if (append_val==0) fp = fopen(write_file, "w");
    else fp = fopen(write_file, "a");

    for (int ix = 0; ix < width; ix++)
    {
      for (int iy = 0; iy < height; iy++)
      {
          fprintf(fp, "%.8f,", mat[ix * width + iy]);
          if ((iy == height - 1) && (ix == width - 1))
              fprintf(fp, "\n");
      }
    }
    fclose(fp);
}

//watch out! stops if it finds 0!
void writeVecCsvInt(int *vec, const char *write_file, int append_val, int vecsize){
    FILE *fp;
    if (append_val==0) fp = fopen(write_file, "w");
    else fp = fopen(write_file, "a");

    for (int i_vec = 0; i_vec < vecsize; i_vec++){
        if (vec[i_vec]==0) {
            fprintf(fp, "\n");
            break;
        }

        fprintf(fp, "%d,",vec[i_vec]);
        if(i_vec==vecsize-1) fprintf(fp, "\n");

    }
    fclose(fp);
}

//watch out! stops if it finds 0!
void writeVecCsvLongLongInt(long long int *vec, const char *write_file, int append_val, int vecsize){
    FILE *fp;
    if (append_val==0) fp = fopen(write_file, "w");
    else fp = fopen(write_file, "a");

    for (int i_vec = 0; i_vec < vecsize; i_vec++){
        if (vec[i_vec]==0) {
            fprintf(fp, "\n");
            break;
        }

        fprintf(fp, "%lld,",vec[i_vec]);
        if(i_vec==vecsize-1) fprintf(fp, "\n");

    }
    fclose(fp);
}

void writeVecCsvFloat(float *vec, const char *write_file, int append_val, int vecsize){
    FILE *fp;
    if (append_val==0) fp = fopen(write_file, "w");
    else fp = fopen(write_file, "a");

    for (int i_vec = 0; i_vec < vecsize; i_vec++){
        if (vec[i_vec] == 0.0f) {
            fprintf(fp, "\n");
            break;
        }

        fprintf(fp, "%f,",vec[i_vec]);
        if(i_vec==vecsize-1) fprintf(fp, "\n");

    }
    fclose(fp);
}

 // Initialize initial concentration with circle in the middle
void initializeArrayCircle ( float *x, float radius_ratio, float val1, float val2 , int width, int height){
    float radius2 = (width/radius_ratio) * (width/radius_ratio);
    for (int ix = 0; ix < width; ix++){
        for (int iy = 0; iy < height; iy++)
        {
            // Distance of point i, j from the origin
            float ds2 = (ix - width/2) * (ix - width/2) + (iy - height/2)*(iy - height/2);
            if (ds2 < radius2)
            {
                x[ix*width + iy] = val2;
            }
            else
            {
                x[ix*width + iy] = val1;
            }
        }
    }
}

// Initialize initial condition of float array
void initializeArrayValInt ( int *x, int val, int width, int height){
    for (int ix = 0; ix < width; ix++){
        for (int iy = 0; iy < height; iy++){
            x[ix*width + iy] = val;
        }
    }
}

void initializeArrayValFloat ( float *x, float val, int width, int height){
    for (int ix = 0; ix < width; ix++){
        for (int iy = 0; iy < height; iy++){
            x[ix*width + iy] = val;
        }
    }
}

void initializeArrayValFloat1D( float *x, float val, int size){
    for (int ix = 0; ix < size; ix++){
            x[ix] = val;
    }
}

// Initialize initial condition of double array
void initializeArrayValDouble ( double *x, double val, int width, int height){
    for (int ix = 0; ix < width; ix++){
        for (int iy = 0; iy < height; iy++){
            x[ix*width + iy] = val;
        }
    }
}

// find max
double max2DArray ( double *x, int width, int height){
    double max = DBL_MIN;
    for (int ix = 0; ix < width; ix++){
        for (int iy = 0; iy < height; iy++){
            if (x[ix*width + iy] > max) max = x[ix*width + iy];
        }
    }
    return max;
}

// find min
double min2DArray ( double *x, int width, int height){
    double min = DBL_MAX;
    for (int ix = 0; ix < width; ix++){
        for (int iy = 0; iy < height; iy++){
            if (x[ix*width + iy] < min) min = x[ix*width + iy];
        }
    }
    return min;
}

void initializeArrayValDoubleRandInt ( double *x, double mean, double delta, int width, int height)
{    
    for (int ix = 0; ix < width; ix++){
        for (int iy = 0; iy < height; iy++){
            x[ix*width + iy] = mean + (( ((double)rand() / (double)(RAND_MAX)) -0.5) * 2.0) * delta;
            if (x[ix*width + iy]<0.0001) x[ix*width + iy]=0.0001;

        }
    }
}

void initializeVecValLongLongInt ( long long int *x, int val, int vecsize){
    for (int i_vec = 0; i_vec < vecsize; i_vec++){
            x[i_vec] = val;
    }
}

void initializeVecValInt (int *x, int val, int vecsize){
    for (int i_vec = 0; i_vec < vecsize; i_vec++){
            x[i_vec] = val;
    }
}

#endif