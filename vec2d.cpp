#include "vec2d.hpp"

vec2d::vec2d(const vector<int> &layer_sizes, bool use_bias_neurons){
  for(int k = 0; k < layer_sizes.size(); k++){
    vector<double> vec1;
    for(int i = 0; i < ((use_bias_neurons && k < (layer_sizes.size() - 1)) ? layer_sizes[k] + 1 : layer_sizes[k]); i++){ // if we add bias, add one artificial node to each layer
      vec1.push_back(0);
    }
    data.push_back(vec1);
  }
}

double& vec2d::operator()(int i, int k){ // returns ith data at kth layer.
  return data[k][i];
}