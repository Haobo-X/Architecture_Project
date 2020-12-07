#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#include <x86intrin.h>
#include <sys/time.h>
#include <pthread.h>
#include "myblockmm.h"

struct thread_info
{
    int tid;
    double **a, **b, **c;
    int array_size;
    int number_of_threads;
    int n;
};
void *mythreaded_vector_blockmm(void *t);

char name[128];
char SID[128];
#define VECTOR_WIDTH 4
void my_threaded_vector_blockmm(double **a, double **b, double **c, int n, int ARRAY_SIZE, int number_of_threads)
{
  int i=0;
  pthread_t *thread;
  struct thread_info *tinfo;
  strcpy(name,"Haobo Xie");
  strcpy(SID,"862188706");
  thread = (pthread_t *)malloc(sizeof(pthread_t)*number_of_threads);
  tinfo = (struct thread_info *)malloc(sizeof(struct thread_info)*number_of_threads);

  for(i = 0 ; i < number_of_threads ; i++)
  {
    tinfo[i].a = a;
    tinfo[i].b = b;
    tinfo[i].c = c;
    tinfo[i].tid = i;
    tinfo[i].number_of_threads = number_of_threads;
    tinfo[i].array_size = ARRAY_SIZE;
    tinfo[i].n = n;
    pthread_create(&thread[i], NULL, mythreaded_vector_blockmm, &tinfo[i]);
  }  
  for(i = 0 ; i < number_of_threads ; i++)
    pthread_join(thread[i], NULL);

  return;
}

#define VECTOR_WIDTH 4
void *mythreaded_vector_blockmm(void *t)
{
  register int i, j, k, ii, jj, kk;
  register __m256d va, vb, vc;
  register __m256d va1, va2, va3, vb1, vb2, vb3, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9;
  register struct thread_info tinfo = *(struct thread_info *)t;
  register int number_of_threads = tinfo.number_of_threads;
  register int tid =  tinfo.tid;
  register double **a = tinfo.a;
  register double **b = tinfo.b;
  register double **c = tinfo.c;
  register int ARRAY_SIZE = tinfo.array_size;
  register int n = tinfo.n;
  // register int n = 64;
  for(i = (ARRAY_SIZE/number_of_threads)*(tid); i < (ARRAY_SIZE/number_of_threads)*(tid+1); i+=ARRAY_SIZE/n)
  {
    for(j = 0; j < ARRAY_SIZE; j+=(ARRAY_SIZE/n))
    {
      for(k = 0; k < ARRAY_SIZE; k+=(ARRAY_SIZE/n))
      {        
        for(ii = i; ii < i+(ARRAY_SIZE/n); ii+=3)
        {
            for(jj = j; jj < j+(ARRAY_SIZE/n); jj+=VECTOR_WIDTH*3)
            {
                vc1 = _mm256_load_pd(&c[ii][jj]);
                vc2 = _mm256_load_pd(&c[ii][jj+VECTOR_WIDTH]);
                vc3 = _mm256_load_pd(&c[ii][jj+VECTOR_WIDTH*2]);
                vc4 = _mm256_load_pd(&c[ii+1][jj]);
                vc5 = _mm256_load_pd(&c[ii+1][jj+VECTOR_WIDTH]);
                vc6 = _mm256_load_pd(&c[ii+1][jj+VECTOR_WIDTH*2]);
                vc7 = _mm256_load_pd(&c[ii+2][jj]);
                vc8 = _mm256_load_pd(&c[ii+2][jj+VECTOR_WIDTH]);
                vc9 = _mm256_load_pd(&c[ii+2][jj+VECTOR_WIDTH*2]);
                    
                for(kk = k; kk < k+(ARRAY_SIZE/n); kk++)
                {
                    va1 = _mm256_broadcast_sd(&a[ii][kk]);
                    va2 = _mm256_broadcast_sd(&a[ii+1][kk]);
                    va3 = _mm256_broadcast_sd(&a[ii+2][kk]);
                    vb1 = _mm256_load_pd(&b[kk][jj]);
                    vb2 = _mm256_load_pd(&b[kk][jj+VECTOR_WIDTH]);
                    vb3 = _mm256_load_pd(&b[kk][jj+VECTOR_WIDTH*2]);
                    vc1 = _mm256_add_pd(vc1,_mm256_mul_pd(va1,vb1));
                    vc2 = _mm256_add_pd(vc2,_mm256_mul_pd(va1,vb2));
                    vc3 = _mm256_add_pd(vc3,_mm256_mul_pd(va1,vb3));
                    vc4 = _mm256_add_pd(vc4,_mm256_mul_pd(va2,vb1));
                    vc5 = _mm256_add_pd(vc5,_mm256_mul_pd(va2,vb2));
                    vc6 = _mm256_add_pd(vc6,_mm256_mul_pd(va2,vb3));
                    vc7 = _mm256_add_pd(vc7,_mm256_mul_pd(va3,vb1));
                    vc8 = _mm256_add_pd(vc8,_mm256_mul_pd(va3,vb2));
                    vc9 = _mm256_add_pd(vc9,_mm256_mul_pd(va3,vb3));
                }
                _mm256_store_pd(&c[ii][jj],vc1);
                _mm256_store_pd(&c[ii][jj+VECTOR_WIDTH],vc2);
                _mm256_store_pd(&c[ii][jj+VECTOR_WIDTH*2],vc3);
                _mm256_store_pd(&c[ii+1][jj],vc4);
                _mm256_store_pd(&c[ii+1][jj+VECTOR_WIDTH],vc5);
                _mm256_store_pd(&c[ii+1][jj+VECTOR_WIDTH*2],vc6);
                _mm256_store_pd(&c[ii+2][jj],vc7);
                _mm256_store_pd(&c[ii+2][jj+VECTOR_WIDTH],vc8);
                _mm256_store_pd(&c[ii+2][jj+VECTOR_WIDTH*2],vc9);
            }
        }
      }
    }
  } 
}

