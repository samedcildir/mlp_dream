#include "vec3d.hpp"
#include "mlp.hpp"
#include "configurations.h"

#include <iostream>

using namespace std;

vec3d::vec3d(const vector<int> &layer_sizes, bool use_random, bool use_bias_neurons){
  for(int k = 0; k < layer_sizes.size() - 1; k++){
    vector<vector<double> > vec1;
    for(int i = 0; i < (use_bias_neurons ? layer_sizes[k] + 1 : layer_sizes[k]); i++){ // if we add bias, add one artificial node to each layer
      vector<double> vec2;
      for(int j = 0; j < layer_sizes[k + 1]; j++){
        if (use_random)
          vec2.push_back(MLP::get_rand()); // when we create real W, we initialize it with random variables
        else
          vec2.push_back(0); // when we create W for keeping last deltaW (used for momentum), we have to initialize it with zeroes
      }
      vec1.push_back(vec2);
    }
    data.push_back(vec1);
  }
}
vec3d::vec3d(const vec3d& vec){
  for(int i = 0; i < vec.data.size(); i++){
    data.push_back(vector<vector<double> >());
    for(int j = 0; j < vec.data[i].size(); j++){
      data[i].push_back(vector<double>(vec.data[i][j]));
    }
  }
}

double& vec3d::operator()(int i, int j, int k){ // returns weigth of edge that connects ith neuron at kth layer to jth neuron at k+1th layer
  return data[k][i][j];
}

ostream& operator<<(ostream &out, const vec3d& v){ // prints weight matrix
  for (int k = 0; k < v.data.size(); k++){
    for (int i = 0; i < v.data[k].size(); i++){
      for (int j = 0; j < v.data[k][i].size(); j++){
        out << "(" << i << ", " << j << ", " << k << "): " << v.data[k][i][j] << endl;
      }
    }
  }

  return out;
}

inline int get_w_idx(int i, int j, int k){
  //return layer_starts_w[k] + i * layer_sizes[k + 1] + j;
#ifdef STRANGE_VERSION
  return layer_starts_w[k] + i + layer_sizes[k]* j;
#else
  return layer_starts_w[k] + i + (layer_sizes[k] + 1) * j;
#endif
}
void vec3d::fill_w_array(float *w){
  for (int k = 0; k < data.size(); k++){
    for (int i = 0; i < data[k].size(); i++){
      for (int j = 0; j < data[k][i].size(); j++){
        w[get_w_idx(i, j, k)] = data[k][i][j];
      }
    }
  }
}
