#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
//gcc -O3 -march=native -fPIC -shared -fopenmp polyphase.c -o libpolyphase.so
//gcc -O3 -march=native -fPIC -shared -fopenmp polyphase.c -o libpolyphase.so

// void get_osamp_polyphase(double * y, double * x, double * h, int64_t half_size, int64_t L, int64_t nx)
// {
//         //naive implementation
//         int64_t N = nx - 2*half_size, M = 2*half_size;
//         int64_t L4 = L & ~3;  
//         #pragma omp parallel for
//         for (int64_t k = 0; k < N; k++)
//         {       
//                 for(int64_t i = 0; i < M; i++)
//                 {
//                         __m256d vx = _mm256_set1_pd(x[k+M-i-1]);
//                         int64_t j = 0;
//                         for(; j < L4; j +=4 )
//                         {
//                                 // y[k*L + j] += x[k+M-i-1] * h[L*i + j];
//                                 __m256d vy = _mm256_loadu_pd(&y[k*L+j]);
//                                 __m256d vh = _mm256_loadu_pd(&h[L*i + j]);
//                                 vy = _mm256_fmadd_pd(vx, vh, vy);
//                                 _mm256_storeu_pd(&y[k*L+j], vy);
//                         }
//                         for(; j < L; j++)
//                         {
//                                 y[k*L + j] += x[k+M-i-1] * h[L*i + j];    
//                         }

//                 }
//         }
// }

void get_osamp_polyphase(double * y, double * x, double * h, int64_t half_size, int64_t L, int64_t nx)
{
        //naive implementation
        int64_t N = nx - 2*half_size, M = 2*half_size;
        #pragma omp parallel for
        for (int64_t k = 0; k < N; k++)
        {       
                for(int64_t i = 0; i < M; i++)
                {
                        for(int64_t j = 0; j < L; j++ )
                        {
                                y[k*L + j] += x[k+M-i-1] * h[L*i + j];
                        }
                }
        }
}
