#include "configurations.h"

int layer_starts_w[layer_sizes_size - 1];

void fill_layer_starts_w(){
  layer_starts_w[0] = 0;
  for (int i = 1; i < layer_sizes_size - 1; i++)
    layer_starts_w[i] = layer_starts_w[i - 1] + (layer_sizes[i - 1] + 1) * layer_sizes[i];
}
