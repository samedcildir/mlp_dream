#ifndef IMAGE_GENERATION_HPP
#define IMAGE_GENERATION_HPP

#include <QObject>
#include <QMutex>
#include "mlp.hpp"
#include "mlp_iterator.h"
#include "saveimage.h"

#include <vector>

using namespace std;

class MainWindow;

class ImageGeneration : public QObject
{
  Q_OBJECT

  float *w[mlp_count];
  unsigned int N;
  uint8_t *res[mlp_count];

  void combine_images(QImage &im);

public:
  ImageGeneration();
  ~ImageGeneration();

  void genarate_image();

  vector<MLP::MLP*> mlps;
  vector<mlp_iterator*> mlp_iters;
  MainWindow *mw;
  SaveImage *si;
  bool draw_samples;

  QMutex mut;


signals:
  void image_ready();

public slots:
  void image_gen_loop();
};

#endif // IMAGE_GENERATION_HPP
