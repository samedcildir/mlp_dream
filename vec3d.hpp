#ifndef VEC3D_HPP
#define VEC3D_HPP

#include <vector>
#include <iostream>

using namespace std;

// 3D vector class for weights. This is not 3D matrix, since the node count on different layers can be different.
class vec3d {
  vector<vector<vector<double> > > data; // data is kept in k,i,j order faster access to weigths in the same layer. Also initialization is easier this way.

public:
  vec3d(const vector<int> &layer_sizes, bool use_random, bool use_bias_neurons);
  vec3d(const vec3d& vec);

  double& operator()(int i, int j, int k); // returns weigth of edge that connects ith neuron at kth layer to jth neuron at k+1th layer

  void fill_w_array(float *w);

  friend ostream& operator<<(ostream &out, const vec3d& v); // for printing weights. After we train a model, we can use it in other programs without training it again.
};

#endif
