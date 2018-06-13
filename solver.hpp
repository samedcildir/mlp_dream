#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <QObject>
#include <QThread>
#include "mlp.hpp"
#include "image_generation.hpp"

#include <vector>
#include "mlp_iterator.h"

using namespace std;

class MainWindow;

class Solver : public QObject
{
  Q_OBJECT

  vector<MLP::MLP*> mlps;
  MainWindow *mw;
  int max_iteration = -1;

public:
  Solver(void);
  ~Solver(void);
  void set_mlps(vector<MLP::MLP*> mlps);
  void set_mainwindow(MainWindow *mw);
  void set_maxiteration(int cnt);
  void genarate_image(QImage &im);

  ImageGeneration im_gen;
  QThread image_gen_thread;
  vector<QThread*> mlp_iter_threads;
  vector<mlp_iterator*> mlp_iters;
  float noise_y_increment;

signals:
  void image_ready();

public slots:
  void solve();
};

#endif // SOLVER_HPP
