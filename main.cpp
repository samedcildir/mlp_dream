#include "mainwindow.h"
#include <QApplication>
#include <QDebug>

#include <cuda.h>
#include <builtin_types.h>

int main(int argc, char *argv[])
{
  int deviceCount = 0;
  int cudaDevice = 0;
  char cudaDeviceName [100];

  cuInit(0);
  cuDeviceGetCount(&deviceCount);
  cuDeviceGet(&cudaDevice, 0);
  cuDeviceGetName(cudaDeviceName, 100, cudaDevice);
  qDebug() << "Number of devices: " << deviceCount;
  qDebug() << "Device name:" << cudaDeviceName;

  QApplication a(argc, argv);
  MainWindow w;
  w.show();

  return a.exec();
}
