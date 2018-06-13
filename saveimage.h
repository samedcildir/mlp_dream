#ifndef SAVEIMAGE_H
#define SAVEIMAGE_H

#include <QObject>
#include "configurations.h"

class MainWindow;
class ImageGeneration;

class SaveImage : public QObject
{
  Q_OBJECT
public:
    SaveImage(ImageGeneration *imgen, MainWindow *mw);

    uint8_t *res[mlp_count];

private:
    ImageGeneration *imgen;
    MainWindow *mw;

    unsigned int N;

    void combine_images(QImage &im);
    void save_image();

signals:
  void image_ready();

public slots:
    void do_stuff();
};

#endif // SAVEIMAGE_H
