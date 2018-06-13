#include "solver.hpp"
#include "mainwindow.h"
#include <fstream>

#include "configurations.h"
#include "mlp.hpp"

#include <QElapsedTimer>

Solver::Solver(void)
{
  noise_y_increment = 0.02f;
}

Solver::~Solver(void){
}

void Solver::set_mlps(vector<MLP::MLP*> mlps){
  this->mlps = mlps;
  im_gen.mlps = mlps;
}

void Solver::set_mainwindow(MainWindow *mw){
  this->mw = mw;
  im_gen.mw = mw;
}

void Solver::set_maxiteration(int cnt){
  max_iteration = cnt;
}

inline void clip(double &x){
  if (x > 1.0) x = 1.0;
  if (x < -1.0) x = -1.0;
}

void Solver::solve(){
#if RAND_MODE == 1
#elif RAND_MODE == 2
  double x[train_sample_count * mlp_count] = { 0 }, y[train_sample_count * mlp_count] = { 0 };
#if BW
  double bw[train_sample_count * mlp_count] = { 0 };
#else
  double r[train_sample_count * mlp_count] = { 0 }, g[train_sample_count * mlp_count] = { 0 }, b[train_sample_count * mlp_count] = { 0 };
#endif
  for (int i = 0; i < train_sample_count * mlp_count; i++){
    x[i] = MLP::get_rand();
    y[i] = MLP::get_rand();
#if BW
    bw[i] = MLP::get_rand();
#else
    r[i] = MLP::get_rand();
    g[i] = MLP::get_rand();
    b[i] = MLP::get_rand();
#endif
  }
#elif RAND_MODE == 3
  QElapsedTimer timer;
  double x[train_sample_count * mlp_count] = { 0 }, y[train_sample_count * mlp_count] = { 0 };
#if BW
  double bw[train_sample_count * mlp_count] = { 0 };
#elif COLOUR_TYPE == 0 || COLOUR_TYPE == 1
  double r[train_sample_count * mlp_count] = { 0 }, g[train_sample_count * mlp_count] = { 0 }, b[train_sample_count * mlp_count] = { 0 };
#endif
  for (int i = 0; i < train_sample_count * mlp_count; i++){
#if BW
    x[i] = MLP::get_smooth_noise(0 + 3*i, 0);
    y[i] = MLP::get_smooth_noise(1 + 3*i, 0);
#if FROM_IMAGE == 0
    bw[i] = MLP::get_smooth_noise(2 + 3*i, 0);
#else
    MLP::get_from_image_bw(x[i], y[i], bw + i);
    clip(bw[i]);
#endif
#elif COLOUR_TYPE == 0 || COLOUR_TYPE == 1
    x[i] = MLP::get_smooth_noise(0 + 5*i, 0);
    y[i] = MLP::get_smooth_noise(1 + 5*i, 0);
#if FROM_IMAGE == 0
    r[i] = MLP::get_smooth_noise(2 + 5*i, 0);
    g[i] = MLP::get_smooth_noise(3 + 5*i, 0);
    b[i] = MLP::get_smooth_noise(4 + 5*i, 0);
#else
    MLP::get_from_image(x[i], y[i], r + i, g + i, b + i);
    clip(r[i]);
    clip(g[i]);
    clip(b[i]);
#endif
#endif
  }
#endif

  for (int i = 0; i < mlp_count; i++){
    mlp_iter_threads.push_back(new QThread());
    mlp_iters.push_back(new mlp_iterator(mlps[i], i));
    mlp_iters[i]->moveToThread(mlp_iter_threads[i]);
    connect(mlp_iter_threads[i], SIGNAL(started()), mlp_iters[i], SLOT(run_iterators()));
    mlp_iter_threads[i]->start();
  }

  QThread::msleep(250);

  im_gen.mlp_iters = mlp_iters;
  connect(&image_gen_thread, SIGNAL(started()), &im_gen, SLOT(image_gen_loop()));
  image_gen_thread.start();
#if RAND_MODE == 3
  timer.start();
#endif

  while(true){
#if RAND_MODE == 3
    quint64 elapsed_time = timer.nsecsElapsed();
    float elapsed_time_f = (elapsed_time / 1000) / 10000.0f;
    timer.restart();

    constexpr float color_change_rate = 1.0f / 4.0f;
    static float y_val = 0.0f;
    y_val += noise_y_increment * elapsed_time_f * 5.0;
#endif
    vector<vector<double> > train_samples_in;
    vector<vector<double> > train_samples_out;

    for (int i = 0; i < train_sample_count * mlp_count; i++){
#if RAND_MODE == 1
      double x = MLP::get_rand();
      double y = MLP::get_rand();
      vector<double> train_sample_in = { x, y };
#elif RAND_MODE == 2
      x[i] += MLP::get_rand() * noise_y_increment;
      y[i] += MLP::get_rand() * noise_y_increment;
      clip(x[i]);
      clip(y[i]);
      vector<double> train_sample_in = { x[i], y[i] };
#elif RAND_MODE == 3
#endif

#if BW
#if RAND_MODE == 1
#if FROM_IMAGE == 0
      double bw = MLP::get_rand();
#else
      double bw;
      MLP::get_from_image_bw(x, y, &bw);
      clip(bw);
#endif
      vector<double> train_sample_out = { bw };
#elif RAND_MODE == 2
#if FROM_IMAGE == 0
      bw[i] += MLP::get_rand() * noise_y_increment;
      clip(bw[i]);
#else
      MLP::get_from_image_bw(x[i], y[i], bw + i);
      clip(bw[i]);
#endif
      vector<double> train_sample_out = { bw[i] };
#elif RAND_MODE == 3
      x[i] = MLP::get_smooth_noise(0 + 3*i, y_val);
      y[i] = MLP::get_smooth_noise(1 + 3*i, y_val);
      clip(x[i]);
      clip(y[i]);
      vector<double> train_sample_in = { x[i], y[i] };

#if FROM_IMAGE == 0
      bw[i] = MLP::get_smooth_noise(2 + 3*i, y_val * color_change_rate); // TODO: change this 4.0
      clip(bw[i]);
#else
      MLP::get_from_image_bw(x[i], y[i], bw + i);
      clip(bw[i]);
#endif

      vector<double> train_sample_out = { bw[i] };
#endif
#else
#if RAND_MODE == 1
#if FROM_IMAGE == 1
      double r,g,b;
      MLP::get_from_image(x, y, &r, &g, &b);
      clip(r);
      clip(g);
      clip(b);
#else
      double r = MLP::get_rand();
      double g = MLP::get_rand();
      double b = MLP::get_rand();
#endif
      vector<double> train_sample_out = { r, g, b };
#elif RAND_MODE == 2
#if FROM_IMAGE == 1
      MLP::get_from_image(x[i], y[i], r + i, g + i, b + i);
      clip(r[i]);
      clip(g[i]);
      clip(b[i]);
#else
      r[i] += MLP::get_rand() * noise_y_increment;
      g[i] += MLP::get_rand() * noise_y_increment;
      b[i] += MLP::get_rand() * noise_y_increment;
      clip(r[i]);
      clip(g[i]);
      clip(b[i]);
#endif
      vector<double> train_sample_out = { r[i], g[i], b[i] };
#elif RAND_MODE == 3
#if COLOUR_TYPE == 0 || COLOUR_TYPE == 1
      x[i] = MLP::get_smooth_noise(0 + 5*i, y_val);
      y[i] = MLP::get_smooth_noise(1 + 5*i, y_val);
      clip(x[i]);
      clip(y[i]);
      vector<double> train_sample_in = { x[i], y[i] };

#if FROM_IMAGE == 0
      r[i] = MLP::get_smooth_noise(2 + 5*i, y_val * color_change_rate);// * 0.01 - 0.99;
      g[i] = MLP::get_smooth_noise(3 + 5*i, y_val * color_change_rate);
      b[i] = MLP::get_smooth_noise(4 + 5*i, y_val * color_change_rate);
      clip(r[i]);
      clip(g[i]);
      clip(b[i]);
      vector<double> train_sample_out = { r[i], g[i], b[i] };
#else
      MLP::get_from_image(x[i], y[i], r + i, g + i, b + i);
      clip(r[i]);
      clip(g[i]);
      clip(b[i]);
      vector<double> train_sample_out = { r[i], g[i], b[i] };
#endif
#endif
#endif
#endif

      train_samples_in.push_back(train_sample_in);
      train_samples_out.push_back(train_sample_out);
    }

    mw->other_mutex.lock();

    for(int i = 0; i < mlp_count; i++)
      mlp_iters[i]->can_it_run = false;
    for(int i = 0; i < mlp_count; i++)
      while (mlp_iters[i]->is_it_running) ;

    MLP::clear_train_samples();
    MLP::add_train_samples(train_samples_in, train_samples_out);

    for(int i = 0; i < mlp_count; i++)
      mlp_iters[i]->can_it_run = true;
    for(int i = 0; i < mlp_count; i++)
      while (!mlp_iters[i]->is_it_running) ;

    mw->other_mutex.unlock();

    // QThread::msleep(50);

    //ofstream off("w.txt");
    //mlp->print_w(off);
  }
}
