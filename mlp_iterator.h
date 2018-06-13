#ifndef MLP_ITERATOR_H
#define MLP_ITERATOR_H

#include <QObject>
#include "mlp.hpp"

class mlp_iterator : public QObject
{
  Q_OBJECT

  MLP::MLP *mlp;
  int idx;

public:
  volatile bool can_it_run;
  volatile bool is_it_running;
  mlp_iterator(MLP::MLP *mlp, int idx);

public slots:
  void run_iterators();
};

#endif // MLP_ITERATOR_H
