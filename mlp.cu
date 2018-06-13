#define CUDA_FILE

#include <stdint.h>
#include "configurations.h"
#include <iostream>

#include "globals.hpp"

using namespace std;

__device__ inline float act_sig(float d){ // 1 / (1 + exp(-x)) // output 0, 1
  return (1.0 / (1.0f + exp(-d)) - 0.5f) * 2.0f;
}
__device__ inline float act_fs(float d){
  return (d / (1.0f + fabsf(d)));
}
__device__ inline float act_sin(float d){
  return sin(d);
}
__device__ inline float act_sinc(float d){
  return sin(d) / d;
}
__device__ inline float act_gauss(float d){
  return (exp(-1.0f * d * d) - 0.5f) * 2.0f;
  //return exp(-1.0f * d * d);
}
__device__ inline float act_relu(float d){
  if (d < 0.0f) return 0.0f;
  else return d;
}
__device__ inline float act_softplus(float d){
    return log(1.0f + exp(d)) - 1.0f;
}

__device__ int layer_sizes_device[layer_sizes_size];
__device__ int layer_starts_w_device[layer_sizes_size - 1];
__device__ int layer_starts_so_device[layer_sizes_size];
__device__ int so_count_for_model_device;
int so_count_for_model;
int w_count_for_model;

#if BW == 0 && COLOUR_TYPE == 1
__device__ inline void convert(uint8_t *colors_in, uint8_t* colors_out)
{
    uint8_t region, remainder, p, q, t;
    uint8_t h = *colors_in;
    uint8_t s = *(colors_in + 1);
    uint8_t v = *(colors_in + 2);
    uint8_t *r = colors_out;
    uint8_t *g = colors_out + 1;
    uint8_t *b = colors_out + 2;

    if (s == 0)
    {
        *r = v;
        *g = v;
        *b = v;
        return;
    }

    region = h / 43;
    remainder = (h - (region * 43)) * 6;

    p = (v * (255 - s)) >> 8;
    q = (v * (255 - ((s * remainder) >> 8))) >> 8;
    t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

    switch (region)
    {
        case 0:
            *r = v; *g = t; *b = p;
            break;
        case 1:
            *r = q; *g = v; *b = p;
            break;
        case 2:
            *r = p; *g = v; *b = t;
            break;
        case 3:
            *r = p; *g = q; *b = v;
            break;
        case 4:
            *r = t; *g = p; *b = v;
            break;
        default:
            *r = v; *g = p; *b = q;
            break;
    }

    return;
}
#elif BW == 0
__device__ inline void convert(uint8_t *colors_in, uint8_t* colors_out) {
  colors_out[0] = colors_in[2];
  colors_out[1] = colors_in[1];
  colors_out[2] = colors_in[0];
}
#elif BW == 1
__device__ inline void convert(uint8_t *colors_in, uint8_t* colors_out) {
  colors_out[0] = colors_in[0];
}
#endif

#if BW
const int res_count_for_model = 1;
#else
const int res_count_for_model = 4; // BGRA ??
#endif

__device__ inline int get_so_idx(int layer_no, int node_no){
  return layer_starts_so_device[layer_no] + node_no;
}
__device__ inline int get_w_idx(int i, int j, int k){
  //return layer_starts_w_device[k] + i * layer_sizes_device[k + 1] + j;
#ifdef STRANGE_VERSION
  return layer_starts_w_device[k] + i + layer_sizes_device[k]* j;
#else
  return layer_starts_w_device[k] + i + (layer_sizes_device[k] + 1) * j;
#endif
}

__device__ inline void get_y(const int idx, const float x_, const float y_, uint8_t* y, /*float* s, */float* o, const float* w){
  //s[0] = x_;
  //s[1] = y_;
  o[0] = x_;
  o[1] = y_;
  uint8_t network_out[NETWORK_MAX_OUTPUT_SIZE] = { 0 };

#if USE_BIAS
  o[2] = 1; // extra neuron for bias
#endif

  int w_idx = get_w_idx(0, 0, 0);
  for (int k = 1; k < layer_sizes_size - 1; k++){
    int o_idx_ = get_so_idx(k - 1, 0);
    int o_idx_2 = get_so_idx(k, 0);
    int lmt = layer_sizes_device[k];
#if USE_BIAS
    int lmt2 = layer_sizes_device[k - 1] + 1; // if we use bias neurons, increase i's limit by one to add bias coefficient
#else
    int lmt2 = layer_sizes_device[k - 1];
#endif
    for (int j = 0; j < lmt; j++){
      float sum = 0;

      int o_idx = o_idx_;
      for(int i = 0; i < lmt2; i++){
        sum += w[w_idx++] * o[o_idx++];
      }
      //s[get_so_idx(k, j)] = sum;
      switch(k){
        case 1:
          o[o_idx_2++] = ACT1(sum); // hidden layer's neurons are nonlinear
          break;
        case 2:
          o[o_idx_2++] = ACT2(sum); // hidden layer's neurons are nonlinear
          break;
        case 3:
          o[o_idx_2++] = ACT3(sum); // hidden layer's neurons are nonlinear
          break;
        case 4:
          o[o_idx_2++] = ACT4(sum); // hidden layer's neurons are nonlinear
          break;
        case 5:
          o[o_idx_2++] = ACT5(sum); // hidden layer's neurons are nonlinear
          break;
        default:
          o[o_idx_2++] = ACTDEF(sum); // hidden layer's neurons are nonlinear
          break;
      }
    }

#if USE_BIAS
    o[o_idx_2] = 1; // add an extra neuron for bias
#endif
  }
  {
    const int k = layer_sizes_size - 1;
    int o_idx_ = get_so_idx(k - 1, 0);
    int lmt = layer_sizes_device[k];
#if USE_BIAS
    int lmt2 = layer_sizes_device[k - 1] + 1; // if we use bias neurons, increase i's limit by one to add bias coefficient
#else
    int lmt2 = layer_sizes_device[k - 1];
#endif
    for (int j = 0; j < lmt; j++){
      float sum = 0;

      int o_idx = o_idx_;
      for(int i = 0; i < lmt2; i++){
        sum += w[w_idx++] * o[o_idx++];
      }

      // s[get_so_idx(k, j)] = sum; // not necessary!!

      float res = ACTLAST(sum);
      // o[get_so_idx(k, j)] = res; // not necessary!!
      int res_int = (res + 1.0f) * 128;
      uint8_t res_uint = res_int;
      if (res_int > 255) res_uint = 255;
      if (res_int <   0) res_uint =   0;

      network_out[j] = res_uint;
    }

    convert(network_out, y);
#if !BW
    y[3] = 0xffu;
#endif
  }
}

__device__ const int width = WIDTH;
__device__ const int height = HEIGHT;
__device__ const float scale_down = my_min(WIDTH, HEIGHT) / 2.0 * SAMPLE_AREA_RATIO;

extern "C"
__global__ void mlpCUDA(uint8_t* res, /*float* s, */float* o, const float* w, int n, int start)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;
    if (ii < n){
      ii += start;

      int x = ii % width;
      int y = ii / width;
      float x_ = (x - width / 2) / scale_down;
      float y_ = (y - height / 2) / scale_down;

      get_y(ii, x_, y_, res + ii * res_count_for_model, /*s + ii * so_count_for_model,*/ o + ii * so_count_for_model_device, w);
    }
}

uint8_t *res_cuda;
float /**s_cuda, */*o_cuda, *w_cuda;
const int stream_count = 1;
cudaStream_t streams[stream_count];
void mlp_run_kernel(uint8_t* res, float* w, int n) {
    int threadsPerBlock = 32 * 4;
    int piece = n / stream_count;
    int blocksPerGrid = (piece + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(w_cuda, w, sizeof(float) * w_count_for_model, cudaMemcpyHostToDevice);

    for(int i = 0; i < stream_count; i++){
      mlpCUDA<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(res_cuda, /*s_cuda,*/ o_cuda, w_cuda, piece, piece * i);

      // load the answer back into the host
      cudaMemcpyAsync(res + sizeof(uint8_t) * piece * i * res_count_for_model, res_cuda + sizeof(uint8_t) * piece * i * res_count_for_model,
                            sizeof(uint8_t) * piece * res_count_for_model, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();
}

__global__ void cuda_init()
{
  layer_starts_so_device[0] = 0;
  layer_starts_w_device[0] = 0;
  for (int i = 1; i < layer_sizes_size; i++){
    layer_starts_so_device[i] = layer_starts_so_device[i - 1] + layer_sizes_device[i - 1] + 1;
  }
  for (int i = 1; i < layer_sizes_size - 1; i++){
    layer_starts_w_device[i] = layer_starts_w_device[i - 1] + (layer_sizes_device[i - 1] + 1) * layer_sizes_device[i];
  }
  so_count_for_model_device = layer_starts_so_device[layer_sizes_size - 1] + layer_sizes_device[layer_sizes_size - 1];
}

void init_kernel(int n) {
  //cout << cudaMemcpy(layer_sizes_device, layer_sizes, sizeof(int) * layer_sizes_size, cudaMemcpyHostToDevice) << endl;
  cout << cudaMemcpyToSymbol(layer_sizes_device, layer_sizes, sizeof(int) * layer_sizes_size, 0, cudaMemcpyHostToDevice) << endl;
  cout << cudaErrorInvalidValue << endl;
  cudaDeviceSynchronize();
  cuda_init << <1, 1 >> >();
  cout << "init?" << endl;
  cudaDeviceSynchronize();
  cout << "init done" << endl;

  so_count_for_model = 0;
  w_count_for_model = 0;
  for (int i = 0; i < layer_sizes_size; i++)
    so_count_for_model += layer_sizes[i] + 1;
  for (int i = 1; i < layer_sizes_size; i++)
    w_count_for_model += (layer_sizes[i - 1] + 1) * layer_sizes[i];

  so_count_for_model--;

  // allocate and copy memory into the device
  cudaMalloc((void **)& res_cuda, sizeof(uint8_t) * n * res_count_for_model);
  //cudaMalloc((void **)& s_cuda, sizeof(float) * n * so_count_for_model);
  cudaMalloc((void **)& o_cuda, sizeof(float) * n * so_count_for_model);
  cudaMalloc((void **)& w_cuda, sizeof(float) * w_count_for_model);

  for(int i = 0; i < stream_count; i++)
    cudaStreamCreate(streams + i);

  cout << "init_kernel done" << endl;
}

void end_kernel() {
    cudaFree(res_cuda);
    //cudaFree(s_cuda);
    cudaFree(o_cuda);
    cudaFree(w_cuda);

    for(int i = 0; i < stream_count; i++)
      cudaStreamDestroy(streams[i]);
}
