#ifndef MLP_HPP
#define MLP_HPP

#include <iostream>
#include <vector>
#include <cmath>    // for sine exp and fabs
#include <fstream>  // for printing outputs to file
#include <sstream>

#include <limits>   // for double max

#include <random>   // for uniform_real_distribution
#include <chrono>   // for seeding random number generator

#include "vec2d.hpp"
#include "vec3d.hpp"
#include "configurations.h"

#include <QImage>

using namespace std;

namespace MLP{
  // PARAMETERS //
  // double initial_rand_max = 0.2; // (-initial_rand_max, initial_rand_max)
  extern int use_nonlinear_output;//
  extern vector<vector<double> > test_samples_in;//
  extern vector<vector<double> > test_samples_out;//
  extern vector<vector<double> > train_samples_in;//
  extern vector<vector<double> > train_samples_out;//
  extern double learning_rate;
  extern double err_threshold;
  extern vector<double> err_threshold_vector;
  extern int print_err_test_at_every;//
  extern int use_bias_neurons;//
  extern int use_momentum;//
  extern double momentum_coeff;//
  extern int use_learning_rate_update;//
  extern double learning_rate_update_increase_coeff;//
  extern double learning_rate_update_decrease_coeff;//
  extern int learning_rate_update_count;//

  // HELPER_FUNCTIONS //
  ostream& operator<<(ostream &out, const vector<double>& v); // prints single dimension vector. Used for printing inputs and outputs since there can be more than one input and output
  vector<double> operator-(const vector<double>& v1, const vector<double>& v2); // subtracts two vector. Used for calculating error (there can be multiple outputs)
  bool operator<(const vector<double>& v1, const vector<double>& v2); // returns true if each element in v1 is smaller than v2. Used for comparing maximum error and threshold error.
  vector<double> vabs(const vector<double>& v); // returns a vector consists of abs of each element in v
  vector<double> vmax(const vector<double>& v1, const vector<double>& v2); // returns a vector consists of maximum elements of v1 and v2.
  double sum(const vector<double> &v); // returns the sum of elements of v

  // MLP CLASS //
  class MLP {
    vec3d last_delta_w; // used for momentum, initialized with zeros
    vec2d s;
    vec2d sigma;
    vec2d o;

  public:
    MLP(const vector<int> &layer_sizes); // constructor
    MLP(const MLP& mlp);
    void calc_s_and_o(const vector<double> &x); // calculates s and o using input sample
    vector<double> get_y(); // call after calc_s_and_o. returns last layer of o which is the output of network.
    vector<double> get_y(const vector<double> &x, vec2d &s, vec2d &o); // creates local s and o so not very efficient
    void learn(const vector<double> &t); // call after calc_s_and_o. teaches network using s,o and t(normal output)
    void print_w(ostream &off); // prints w matrix of this network to output stream
    void iterate(int mlp_idx);

    void fill_w_array(float *w);

    vector<int> layer_sizes;
    vec3d w;
  };

  // NONLINEAR_FUNCTIONS //
  double act_sig(double d); // 1 / (1 + exp(-x)) // output 0, 1
  double act_sigt(double d); // https://en.wikipedia.org/wiki/Activation_function
  double act_fs(double d); // x / (1 + |x|)) // output -1, 1
  double act_fst(double d); // https://en.wikipedia.org/wiki/Activation_function
  double act_sin(double d);
  double act_sint(double d);
  double act_sinc(double d);
  double act_sinct(double d);
  double act_gauss(double d);
  double act_gausst(double d);
  double act_relu(double d);
  double act_relut(double d);
  double act_softplus(double d);
  double act_softplust(double d);

  // TESTS //
  bool test_with_train_set(MLP &mlp, bool test_max_or_mean, ofstream *off);

  // WORST_N //
  vector<int> find_worst_n_idx(int n, MLP &mlp);

  // PRINT RESULTS //
  void print_results(MLP &mlp, ostream &off);
  void draw_samples(QImage &im, vector<vector<double> >& in, vector<vector<double> >& out);

  // RANDOM // random number generator for initial weights
  void init_rand(double initial_rand_max);
  double get_rand();
  float get_smooth_noise(int i, float y);
  extern QImage input_image;
  void set_image(QImage &in_img);
  void get_from_image(double x, double y, double* r, double* g, double* b);
  void get_from_image_bw(double x, double y, double *bw);

  void add_train_samples(vector<vector<double> >& in, vector<vector<double> >& out);
  void add_test_samples(vector<vector<double> >& in, vector<vector<double> >& out);
  void clear_train_samples();
  void clear_test_samples();
  void enable_momentum(double coeff);
  void disable_momentum();
  void enable_bias();
  void disable_bias();
  void set_nonlinear_output(bool type);
  void enable_learning_rate_update(double increase_coeff, double decrease_coeff, int count);
  void disable_learning_rate_update();
  void set_print_err_test_at_every(int n);
  void set_image_at_every(int n);
  MLP getMLP(vector<int> layer_sizes, double learning_rate, double err_threshold, bool use_bias_neurons);
}

#endif
