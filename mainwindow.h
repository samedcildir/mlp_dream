#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QMutex>

#include "mlp.hpp"
#include "solver.hpp"
#include "configurations.h"

class SaveImage;

using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

  QThread thread;
  QThread thread2;
  Solver solver;
  vector<MLP::MLP*> mlps;
  int fps = 0;

  void update_fps();

public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

  Ui::MainWindow *ui;
  SaveImage *si;

  const int im_width = WIDTH;
  const int im_height = HEIGHT;
  QImage *image;

  QMutex image_mutex;
  QMutex other_mutex;

  vector<int> layer_sizes_mainwindow;

signals:
  void start_process();

public slots:
  void image_ready();

private slots:
  void on_pushButton_clicked();

  void on_horizontalScrollBar_valueChanged(int value);

  void on_checkBox_toggled(bool checked);
};

#endif // MAINWINDOW_H
