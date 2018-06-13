#include "mlp_iterator.h"

mlp_iterator::mlp_iterator(MLP::MLP *mlp, int idx)
{
  this->mlp = mlp;
  this->idx = idx;
  can_it_run = true;
  is_it_running = false;
}

void mlp_iterator::run_iterators(){
  std::cout << "run iter: " << idx << std::endl;
  while(true){
    if (can_it_run){
      is_it_running = true;
      mlp->iterate(idx);
    }
    else {
      is_it_running = false;
    }
  }
}
