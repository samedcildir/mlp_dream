#include "saveimage.h"
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "configurations.h"
#include "image_generation.hpp"

#include "globals.hpp"

SaveImage::SaveImage(ImageGeneration *imgen, MainWindow *mw)
{
    this->imgen = imgen;
    this->mw = mw;

    N = WIDTH * HEIGHT;

  #if BW
    for(int i = 0; i < mlp_count; i++)
      res[i] = new uint8_t[N];
  #else
    for(int i = 0; i < mlp_count; i++)
      res[i] = new uint8_t[N * 4];
  #endif
}

#if BW
    QImage im(WIDTH, HEIGHT, QImage::Format_Grayscale8);
#else
    QImage im(WIDTH, HEIGHT, QImage::Format_RGB32);
#endif

void SaveImage::do_stuff(){
    imgen->mut.lock();
    combine_images(im);
    imgen->mut.unlock();

    mw->image_mutex.lock();
    delete mw->image;
    mw->image = new QImage(im);
    mw->image_mutex.unlock();
    emit image_ready();

    if (mw->ui->checkBox_2->isChecked())
        save_image();
}

void SaveImage::combine_images(QImage &im){
#if BW
  uint8_t *rowData = (uint8_t*)im.scanLine(0);

#if COMBINE_MODE == 1
  for(int i = 0; i < N; i++){
    int sum = 0;
    for(int j = 0; j < mlp_count; j++)
      sum += res[j][i];
    rowData[i] = sum / mlp_count;
  }
#elif COMBINE_MODE == 2
  for(int i = 0; i < N; i++){
    int max = 0;
    for(int j = 0; j < mlp_count; j++)
      if (res[j][i] > max) max = res[j][i];
    rowData[i] = max;
  }
#elif COMBINE_MODE == 3
  for(int i = 0; i < N; i++){
    int min = INT32_MAX;
    for(int j = 0; j < mlp_count; j++)
      if (res[j][i] < min) min = res[j][i];
    rowData[i] = min;
  }
#elif COMBINE_MODE == 4
  for(int i = 0; i < N; i++){
    double res_ = 1.0;
    for(int j = 0; j < mlp_count; j++)
      res_ *= res[j][i] / 256.0;
    rowData[i] = res_ * 256.0;
  }
#elif COMBINE_MODE == 5
  for(int i = 0; i < N; i++){
    double res_ = res[0][i] / 256.0;
    for(int j = 1; j < mlp_count; j++){
        if (res_ < 0.5)
            res_ = 2 * res_ * res[j][i] / 256.0;
        else
            res_ = 1.0 - 2 * (1.0 - res[j][i] / 256.0) * (1.0 - res_);
    }
    rowData[i] = res_ * 256.0;
  }
#elif COMBINE_MODE == 6
  static PerlinNoise pn;
  static int im_idx = 0;
  im_idx++;

  for(int i = 0; i < N; i++){
    double x = i % WIDTH;
    double y = i / WIDTH;
    double noise = pn.noise(x / WIDTH, y / HEIGHT, im_idx * 0.01);

    // rowData[i] = noise * 256;
    rowData[i] = noise * res[0][i] + (1.0 - noise) * res[1][i];
  }
#endif
#else
  uint8_t *rowData = (uint8_t*)im.scanLine(0);

#if COMBINE_MODE == 1
  for(int i = 0; i < N * 4; i++){
    int sum = 0;
    for(int j = 0; j < mlp_count; j++)
      sum += res[j][i];
    rowData[i] = sum / mlp_count;
  }
#elif COMBINE_MODE == 2
  for(int i = 0; i < N * 4; i++){
    int max = 0;
    for(int j = 0; j < mlp_count; j++)
      if (res[j][i] > max) max = res[j][i];
    rowData[i] = max;
  }
#elif COMBINE_MODE == 3
  for(int i = 0; i < N * 4; i++){
    int min = INT32_MAX;
    for(int j = 0; j < mlp_count; j++)
      if (res[j][i] < min) min = res[j][i];
    rowData[i] = min;
  }
#elif COMBINE_MODE == 4
  for(int i = 0; i < N * 4; i++){
    double res_ = 1.0;
    for(int j = 0; j < mlp_count; j++)
      res_ *= res[j][i] / 256.0;
    rowData[i] = res_ * 256.0;
  }
#elif COMBINE_MODE == 5
  for(int i = 0; i < N * 4; i++){
    double res_ = res[0][i] / 256.0;
    for(int j = 1; j < mlp_count; j++){
        if (res_ < 0.5)
            res_ = 2 * res_ * res[j][i] / 256.0;
        else
            res_ = 1.0 - 2 * (1.0 - res[j][i] / 256.0) * (1.0 - res_);
    }
    rowData[i] = res_ * 256.0;
  }
#elif COMBINE_MODE == 6
  static PerlinNoise pn;
  static int im_idx = 0;
  im_idx++;

  for(int i = 0; i < N * 4; i++){
    int i_ = i / 4;
    double x = i_ % WIDTH;
    double y = i_ / WIDTH;
    double noise = pn.noise(x / WIDTH, y / HEIGHT, im_idx * 0.01);

    // rowData[i] = noise * 256;
    rowData[i] = noise * res[0][i] + (1.0 - noise) * res[1][i];
  }
#endif
#endif

  if (imgen->draw_samples){
    mw->other_mutex.lock();
    vector<vector<double> > samples_in(MLP::train_samples_in);
    vector<vector<double> > samples_out(MLP::train_samples_out);
    MLP::draw_samples(im, samples_in, samples_out);
    mw->other_mutex.unlock();
  }
}

void SaveImage::save_image()
{
  mw->image_mutex.lock();
  if (mw->image != NULL){
    save_cnt++;
#if RESCALE_WHEN_SAVING
    mw->image->scaled(SAVE_WIDTH, SAVE_HEIGHT, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).save(QString("%1_%2.bmp").arg(file_prefix).arg(cnt));
#else
    mw->image->save(QString("%1_%2.bmp").arg(save_file_prefix).arg(save_cnt, 6, 10, QChar('0')));
#endif
  }
  mw->image_mutex.unlock();
}
