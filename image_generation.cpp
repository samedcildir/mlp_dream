#include "image_generation.hpp"

#include "configurations.h"
#include "mlp.hpp"

#include <cuda.h>
#include <builtin_types.h>
#include "mainwindow.h"
#include "perlinnoise.h"

#include <QElapsedTimer>

// Forward declare the function in the .cu file
void mlp_run_kernel(uint8_t* res, float* w, int n);
void init_kernel(int n);
void end_kernel();

ImageGeneration::ImageGeneration()
{
  draw_samples = false;
  N = WIDTH * HEIGHT;

  int w_count_for_model = 0;
  for (int i = 1; i < layer_sizes_size; i++)
    w_count_for_model += (layer_sizes[i - 1] + 1) * layer_sizes[i];

#if BW
  for(int i = 0; i < mlp_count; i++)
    res[i] = new uint8_t[N];
#else
  for(int i = 0; i < mlp_count; i++)
    res[i] = new uint8_t[N * 4];
#endif
  for(int i = 0; i < mlp_count; i++)
    w[i] = new float[w_count_for_model];

  init_kernel(N);
}

ImageGeneration::~ImageGeneration(){
  end_kernel();
}

QElapsedTimer qet;
qint64 tm0 = 0, tm1 = 0, tm2 = 0, tm3 = 0, tm4 = 0, tm5 = 0;
void ImageGeneration::genarate_image(){
  mw->other_mutex.lock();
  for(int i = 0; i < mlp_count; i++)
    mlp_iters[i]->can_it_run = false;
  for(int i = 0; i < mlp_count; i++)
    while (mlp_iters[i]->is_it_running) ;

  for(int i = 0; i < mlp_count; i++)
    mlps[i]->fill_w_array(w[i]);

  for(int i = 0; i < mlp_count; i++)
    mlp_iters[i]->can_it_run = true;
  for(int i = 0; i < mlp_count; i++)
    while (!mlp_iters[i]->is_it_running) ;
  mw->other_mutex.unlock();

  tm1 = qet.nsecsElapsed();

  for(int i = 0; i < mlp_count; i++)
    mlp_run_kernel(res[i], w[i], N);

  tm2 = qet.nsecsElapsed();

  tm3 = qet.nsecsElapsed();
  tm4 = qet.nsecsElapsed();
}

void ImageGeneration::image_gen_loop(){
  cout << "image_gen_loop START" << endl;

  qet.start();
  while(true){
    qet.restart();

    tm0 = qet.nsecsElapsed();

    genarate_image();

    mut.lock();
    for(int i = 0; i < mlp_count; i++)
#if BW
        memcpy(si->res[i], res[i], N);
#else
        memcpy(si->res[i], res[i], N * 4);
#endif
    mut.unlock();

    emit image_ready();

    tm5 = qet.nsecsElapsed();

    //ofstream off("w.txt");
    //mlp->print_w(off);

    /// cout << tm0 << " - " << (tm1 - tm0) << " - " << (tm2 - tm1) << " - " << (tm3 - tm2) << " - " << (tm4 - tm3) << " - " << (tm5 - tm4) << endl;
  }
}
