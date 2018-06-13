#ifndef VEC2D_HPP
#define VEC2D_HPP

#include <vector>
#include <iostream>

using namespace std;

// 2D vector class for sigma, s and o. This is not 2D matrix since the node count on different layers can be different.
class vec2d {
  vector<vector<double> > data; // data is kept in k,i order

public:
  vec2d(const vector<int> &layer_sizes, bool use_bias_neurons);

  double& operator()(int i, int k); // returns ith data at kth layer.
};

#endif
