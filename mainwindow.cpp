#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "saveimage.h"

#include "globals.hpp"

#include <QTime>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
  fill_layer_starts_w();

  ui->setupUi(this);
  ui->horizontalScrollBar->setMaximum(50);
  ui->horizontalScrollBar->setMinimum(1);
  ui->horizontalScrollBar->setPageStep(1);
  ui->horizontalScrollBar->setValue(solver.noise_y_increment * 500);

  ui->checkBox->setChecked(solver.im_gen.draw_samples);

  MLP::init_rand(1.0);

  //im_width = ui->main_image->width();
  //im_height = ui->main_image->height();

  constexpr double err_threshold = 0.1;
#if USE_BIAS
  bool use_bias = true;
#else
  bool use_bias = false;
#endif
  for (int i = 0; i < layer_sizes_size; i++) layer_sizes_mainwindow.push_back(layer_sizes[i]);
  for (int i = 0; i < mlp_count; i++)
    mlps.push_back(new MLP::MLP(MLP::getMLP(layer_sizes_mainwindow, 0.0005, err_threshold, use_bias))); // initialize MLP objects

#if FROM_IMAGE
  QImage in_img(inout_file_name);
  MLP::set_image(in_img);
#endif

  MLP::enable_momentum(0.03);
  MLP::set_nonlinear_output(true);
#if LEARNING_RATE_UPDATE == 0
  MLP::disable_learning_rate_update();
#else
  MLP::enable_learning_rate_update(0.005, 0.002, 3); // inc - dec - count
#endif
  MLP::set_print_err_test_at_every(1024);

  solver.im_gen.moveToThread(&solver.image_gen_thread);
  solver.moveToThread(&thread);
  connect(this, SIGNAL(start_process()), &solver, SLOT(solve()));
  solver.set_mlps(mlps);
  solver.set_mainwindow(this);
  solver.set_maxiteration(16);
  //solver.set_maxiteration(256);
  thread.start();
#if BW
  image = new QImage(QSize(im_width, im_height), QImage::Format_Grayscale8);
#else
  image = new QImage(QSize(im_width, im_height), QImage::Format_RGB32);
#endif

  si = new SaveImage(&solver.im_gen, this);
  solver.im_gen.si = si;
  si->moveToThread(&thread2);
  connect(&(solver.im_gen), SIGNAL(image_ready()), si, SLOT(do_stuff()));
  connect(si, SIGNAL(image_ready()), this, SLOT(image_ready()));
  thread2.start();

  emit start_process();
}

MainWindow::~MainWindow()
{
  delete ui;
  for (int i = 0; i < mlp_count; i++)
    delete mlps[i];
  delete image;
}

QTime fps_timer;
unsigned int frames = 0;
void MainWindow::update_fps(){
  if (fps_timer.elapsed() == 0){
    fps_timer.start();
    return;
  }

  frames++;
  if (fps_timer.elapsed() > 1000){
    fps_timer.restart();
    fps = frames;
    frames = 0;
  }
}
void MainWindow::image_ready(){
  update_fps();

  setWindowTitle(QString("MLP Image Generator - FPS: %1 - NoiseIncrement: %2").arg(fps).arg(solver.noise_y_increment));
  image_mutex.lock();

  int w = ui->main_image->width();
  int h = ui->main_image->height();

#if USE_SMOOTHNESS_WHILE_RESCALING_FOR_SCREEN
  ui->main_image->setPixmap(QPixmap::fromImage(*image).scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation));
#else
  ui->main_image->setPixmap(QPixmap::fromImage(*image).scaled(w, h, Qt::KeepAspectRatio));
#endif

  image_mutex.unlock();
}

void MainWindow::on_pushButton_clicked()
{
  if (ui->checkBox_2->isChecked()) return;

  image_mutex.lock();
  if (image != NULL){
      save_cnt++;
  #if RESCALE_WHEN_SAVING
      image->scaled(SAVE_WIDTH, SAVE_HEIGHT, Qt::IgnoreAspectRatio, Qt::SmoothTransformation).save(QString("%1_%2.bmp").arg(file_prefix).arg(cnt));
  #else
      image->save(QString("%1_%2.bmp").arg(save_file_prefix).arg(save_cnt, 6, 10, QChar('0')));
  #endif
  }
  image_mutex.unlock();
}

void MainWindow::on_horizontalScrollBar_valueChanged(int value)
{
  solver.noise_y_increment = value / 500.0f;
}

void MainWindow::on_checkBox_toggled(bool checked)
{
  solver.im_gen.draw_samples = checked;
}
