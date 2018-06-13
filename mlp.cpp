// g++ -o mlp.exe mlp.cpp --std=c++11

#include "mlp.hpp"
#include "configurations.h"
#include "mainwindow.h"
#include "FastNoise.h"
#include "perlinnoise.h"

#include "globals.hpp"

using namespace std;

namespace MLP{
  #pragma region PARAMETERS
  int use_nonlinear_output = 0;
  vector<vector<double> > test_samples_in;
  vector<vector<double> > test_samples_out;
  vector<vector<double> > train_samples_in;
  vector<vector<double> > train_samples_out;
  double learning_rate;
  double err_threshold;
  vector<double> err_threshold_vector;
  int print_err_test_at_every = 64;
  int use_bias_neurons = 0;
  int use_momentum = 0;
  double momentum_coeff;
  int use_learning_rate_update = 0;
  double learning_rate_update_increase_coeff;
  double learning_rate_update_decrease_coeff;
  int learning_rate_update_count;
  #pragma endregion

  #pragma region HELPER_FUNCTIONS
  ostream& operator<<(ostream &out, const vector<double>& v){ // prints single dimension vector. Used for printing inputs and outputs since there can be more than one input and output
    out << "(";
    for (int i = 0; i < v.size(); i++){
      out << v[i];
      if (i != v.size() - 1) out << ", ";
    }
    out << ")";

    return out;
  }

  vector<double> operator-(const vector<double>& v1, const vector<double>& v2){ // subtracts two vector. Used for calculating error (there can be multiple outputs)
    vector<double> res;

    for (int i = 0; i < v1.size(); i++){
      res.push_back(v1[i] - v2[i]);
    }

    return res;
  }

  bool operator<(const vector<double>& v1, const vector<double>& v2){ // returns true if each element in v1 is smaller than v2. Used for comparing maximum error and threshold error.
    for (int i = 0; i < v1.size(); i++)
      if(v1[i] >= v2[i]) return false;

    return true;
  }

  vector<double> vabs(const vector<double>& v){ // returns a vector consists of abs of each element in v
    vector<double> res;

    for (int i = 0; i < v.size(); i++){
      double e = v[i];
      res.push_back(e < 0 ? -e : e);
    }

    return res;
  }

  double max(double v1, double v2){
    return v1 < v2 ? v2 : v1;
  }
  vector<double> vmax(const vector<double>& v1, const vector<double>& v2){ // returns a vector consists of maximum elements of v1 and v2.
    vector<double> res;

    for (int i = 0; i < v1.size(); i++){
      res.push_back(max(v1[i], v2[i]));
    }

    return res;
  }

  double sum(const vector<double> &v){ // returns the sum of elements of v
    double res = 0;

    for (int i = 0; i < v.size(); i++) res += v[i];

    return res;
  }
  #pragma endregion

  #pragma region MLP_CLASS
  MLP::MLP(const vector<int> &layer_sizes):
                      w(layer_sizes, true, use_bias_neurons),last_delta_w(layer_sizes, false, use_bias_neurons)
                      ,s(layer_sizes, use_bias_neurons),layer_sizes(layer_sizes)
                      ,sigma(layer_sizes, use_bias_neurons),o(layer_sizes, use_bias_neurons) { } // constructor
  MLP::MLP(const MLP& mlp):w(mlp.layer_sizes, true, use_bias_neurons), last_delta_w(mlp.layer_sizes, false, use_bias_neurons)
                      ,s(mlp.layer_sizes, use_bias_neurons), layer_sizes(mlp.layer_sizes)
                      ,sigma(mlp.layer_sizes, use_bias_neurons), o(mlp.layer_sizes, use_bias_neurons) {}

  void MLP::calc_s_and_o(const vector<double> &x){ // calculates s and o using input sample
    for (int i = 0; i < x.size(); i++){
      s(i, 0) = x[i];
      o(i, 0) = x[i]; // input neurons are not nonlinear
    }
    if (use_bias_neurons) o(x.size(), 0) = 1; // add an extra neuron for bias
    for (int k = 1; k < layer_sizes.size() - 1; k++){
      for (int j = 0; j < layer_sizes[k]; j++){
        double sum = 0;
        for(int i = 0; i < (use_bias_neurons ? layer_sizes[k - 1] + 1 : layer_sizes[k - 1]); i++){ // if we use bias neurons, increase i's limit by one to add bias coefficient
          sum += w(i, j, k - 1) * o(i, k - 1);
        }
        s(j, k) = sum;
        if (k == 1)
          o(j, k) = ACT1(sum);
        else if (k == 2)
          o(j, k) = ACT2(sum);
        else if (k == 3)
          o(j, k) = ACT3(sum);
        else if (k == 4)
          o(j, k) = ACT4(sum);
        else if (k == 5)
          o(j, k) = ACT5(sum);
        else
          o(j, k) = ACTDEF(sum);
      }
      if (use_bias_neurons) o(layer_sizes[k], k) = 1; // add an extra neuron for bias
    }
    {
      int k = layer_sizes.size() - 1;
      for (int j = 0; j < layer_sizes[k]; j++){
        double sum = 0;
        for(int i = 0; i < (use_bias_neurons ? layer_sizes[k - 1] + 1 : layer_sizes[k - 1]); i++){
          sum += w(i, j, k - 1) * o(i, k - 1);
        }
        s(j, k) = sum;
        o(j, k) = ACTLAST(sum);
      }
    }
  }

  vector<double> MLP::get_y(){ // call after calc_s_and_o. returns last layer of o which is the output of network.
    vector<double> y;
    for(int i = 0; i < layer_sizes[layer_sizes.size() - 1]; i++)
      y.push_back(o(i, layer_sizes.size() - 1));
    return y;
  }

  vector<double> MLP::get_y(const vector<double> &x, vec2d &s, vec2d &o){
    for (int i = 0; i < x.size(); i++){
      s(i, 0) = x[i];
      o(i, 0) = x[i]; // input neurons are not nonlinear
    }
    if (use_bias_neurons) o(x.size(), 0) = 1; // add an extra neuron for bias
    for (int k = 1; k < layer_sizes.size() - 1; k++){
      for (int j = 0; j < layer_sizes[k]; j++){
        double sum = 0;
        for(int i = 0; i < (use_bias_neurons ? layer_sizes[k - 1] + 1 : layer_sizes[k - 1]); i++){ // if we use bias neurons, increase i's limit by one to add bias coefficient
          sum += w(i, j, k - 1) * o(i, k - 1);
        }
        s(j, k) = sum;
        if (k == 1)
          o(j, k) = ACT1(sum);
        else if (k == 2)
          o(j, k) = ACT2(sum);
        else if (k == 3)
          o(j, k) = ACT3(sum);
        else if (k == 4)
          o(j, k) = ACT4(sum);
        else if (k == 5)
          o(j, k) = ACT5(sum);
        else
          o(j, k) = ACTDEF(sum);
      }
      if (use_bias_neurons) o(layer_sizes[k], k) = 1; // add an extra neuron for bias
    }
    {
      int k = layer_sizes.size() - 1;
      for (int j = 0; j < layer_sizes[k]; j++){
        double sum = 0;
        for(int i = 0; i < (use_bias_neurons ? layer_sizes[k - 1] + 1 : layer_sizes[k - 1]); i++){
          sum += w(i, j, k - 1) * o(i, k - 1);
        }
        s(j, k) = sum;
        o(j, k) = ACTLAST(sum);
      }
    }

    vector<double> y;
    for(int i = 0; i < layer_sizes[layer_sizes.size() - 1]; i++)
      y.push_back(o(i, layer_sizes.size() - 1));
    return y;
  }

  void MLP::learn(const vector<double> &t){ // call after calc_s_and_o. teaches network using s,o and t(normal output)
    for (int j = 0; j < layer_sizes[layer_sizes.size() - 1]; j++) { // for output layer
      sigma(j, layer_sizes.size() - 1) = (t[j] - o(j, layer_sizes.size() - 1)) * ACTLASTt(s(j, layer_sizes.size() - 1));

      for (int i = 0; i < (use_bias_neurons ? layer_sizes[layer_sizes.size() - 2] + 1 : layer_sizes[layer_sizes.size() - 2]); i++) {
        if (use_momentum){
          double delta_w = learning_rate * sigma(j, layer_sizes.size() - 1) * o(i, layer_sizes.size() - 2); // new delta_w
          last_delta_w(i, j, layer_sizes.size() - 2) = last_delta_w(i, j, layer_sizes.size() - 2) * momentum_coeff + delta_w; // momentum added delta_w
          w(i, j, layer_sizes.size() - 2) += last_delta_w(i, j, layer_sizes.size() - 2); // momentum added delta_w
        }
        else
          w(i, j, layer_sizes.size() - 2) += learning_rate * sigma(j, layer_sizes.size() - 1) * o(i, layer_sizes.size() - 2);
      }
    }

    for (int k = layer_sizes.size() - 3; k >= 0; k--) { // for other layers
      for (int j = 0; j < layer_sizes[k + 1]; j++) {
        double sum = 0;
        for (int l = 0; l < layer_sizes[k + 2]; l++) {
          sum += sigma(l, k + 2) * w(j, l, k + 1);
        }
        if (k == 0)
          sigma(j, k + 1) = sum * ACT1t(s(j, k + 1));
        else if (k == 1)
          sigma(j, k + 1) = sum * ACT2t(s(j, k + 1));
        else if (k == 2)
          sigma(j, k + 1) = sum * ACT3t(s(j, k + 1));
        else if (k == 3)
          sigma(j, k + 1) = sum * ACT4t(s(j, k + 1));
        else if (k == 4)
          sigma(j, k + 1) = sum * ACT5t(s(j, k + 1));
        else
          sigma(j, k + 1) = sum * ACTDEFt(s(j, k + 1));

        for (int i = 0; i < (use_bias_neurons ? layer_sizes[k] + 1 : layer_sizes[k]); i++) {
          if (use_momentum){
            double delta_w = learning_rate * sigma(j, k + 1) * o(i, k); // new delta_w
            last_delta_w(i, j, k) = last_delta_w(i, j, k) * momentum_coeff + delta_w; // momentum added delta_w
            w(i, j, k) += last_delta_w(i, j, k); // momentum added delta_w
          }
          else
            w(i, j, k) += learning_rate * sigma(j, k + 1) * o(i, k);
        }
      }
    }
  }
  void MLP::iterate(int mlp_idx){
    for(int j = train_samples_in.size() / mlp_count * mlp_idx; j < train_samples_in.size() / mlp_count * (mlp_idx + 1); j++){ // iterates over each training sample
      calc_s_and_o(train_samples_in[j]);
      learn(train_samples_out[j]);
    }
    //cout << "it ended: " << mlp_idx << endl;
  }

  void MLP::print_w(ostream &off){ // prints w matrix of this network to output stream
    for (int i = 0; i < layer_sizes.size(); i++)
      off << layer_sizes[i] << " ";
    off << endl;
    off << w << endl;
  }

  void MLP::fill_w_array(float *w){
    this->w.fill_w_array(w);
  }
  #pragma endregion

  #pragma region NONLINEAR_FUNCTIONS
  double act_sig(double d){ // 1 / (1 + exp(-x)) // output 0, 1
    return (1.0 / (1.0 + exp(-d)) - 0.5) * 2.0;
  }
  double act_sigt(double d){
    return 2.0 * act_sig(d) * (1 - act_sig(d)); // https://en.wikipedia.org/wiki/Activation_function
  }

  double act_fs(double d){ // x / (1 + |x|)) // output -1, 1
    return (d / (1.0 + fabs(d)));
  }
  double act_fst(double d){
    return 1.0 / ((1.0 + fabs(d)) * (1.0 + fabs(d))); // https://en.wikipedia.org/wiki/Activation_function
  }

  double act_sin(double d){
    return sin(d);
  }
  double act_sint(double d){
    return cos(d);
  }

  double act_sinc(double d){
    if (d == 0) return 1;
    return sin(d) / d;
  }
  double act_sinct(double d){
    if (d == 0) return 0;
    return cos(d) / d - sin(d) / (d * d);
  }

  double act_gauss(double d){
    return (exp(-1.0 * d * d) - 0.5) * 2.0;
  }
  double act_gausst(double d){
    return -4 * d * exp(-1.0 * d * d);
  }

  double act_relu(double d){
      if (d < 0.0) return 0.0;
      else return d;
  }
  double act_relut(double d){
      if (d < 0.0) return 0.0;
      else return 1.0;
  }

  double act_softplus(double d){
      return log(1.0 + exp(d)) - 1.0;
  }
  double act_softplust(double d){
      return 1.0 / (1.0 + exp(-d));
  }
  #pragma endregion

  #pragma region TESTS
  bool test_with_train_set(MLP &mlp, bool test_max_or_mean, ofstream *off){
    static int iter_cnt = 0; // iteration count
    static double last_energy = std::numeric_limits<double>::max(); // initalize last energy to maximum for not decreasing learning rate at first
    static int last_n_energy_descending = 0; // keeps track of last energy function descending count

    iter_cnt++;
    double err = 0;
    vector<double> max_err;
    for(int i = 0; i < train_samples_out[0].size(); i++) max_err.push_back(0);

    for(int i = 0; i < train_samples_in.size(); i++){
      mlp.calc_s_and_o(train_samples_in[i]); // calculate o
      vector<double> y = mlp.get_y(); // get y (last layer of o)
      for(int j = 0; j < y.size(); j++)
        err += (train_samples_out[i][j] - y[j]) * (train_samples_out[i][j] - y[j]) / 2; // calculate error
      max_err = vmax(max_err, vabs(train_samples_out[i] - y)); // calculate max_err
    }

    err /=  train_samples_in.size(); // takes average of total error

    if (off != NULL)
      (*off) << err << " ";

    /// LEARNING_rate_UPDATE ///
    if (use_learning_rate_update){ // updates learning_rate adaptively
      if (last_energy > err) { // if error is descending
        last_n_energy_descending++; // increase count by one
        if (last_n_energy_descending >= learning_rate_update_count){ // if count is bigger than threshold
          last_n_energy_descending = 0; // restart count
          learning_rate += learning_rate_update_increase_coeff * learning_rate; // update learning_rate
        }
      }
      else if (last_energy < err){ // if error is increasing
        last_n_energy_descending = 0; // restart count
        learning_rate -= learning_rate_update_decrease_coeff * learning_rate; // update learning_rate
      }
      last_energy = err; // update last_energy
    }
    ///                             ///

    if (iter_cnt % print_err_test_at_every == 0){
      cout << "err: " << err << "\tmax_err: " << max_err << "\tlearning_rate: " << learning_rate << endl; // print error statistics in every {print_err_test_at_every} iteration
    }

    if (test_max_or_mean)
      return max_err < err_threshold_vector;
    else
      return err < err_threshold;
  }
  #pragma endregion

  #pragma region WORST_N
  vector<int> find_worst_n_idx(int n, MLP &mlp){ // returns index of worst n training samples
    vector<int> res;
    vector<double> max_errs;
    for(int i = 0; i < n; i++) res.push_back(0);
    for(int i = 0; i < n; i++) max_errs.push_back(0);

    for(int i = 0; i < train_samples_in.size(); i++){
      vector<double> y;
      mlp.calc_s_and_o(train_samples_in[i]);
      y = mlp.get_y();

      vector<double> err = vabs(train_samples_out[i] - y);
      double sum_err = sum(err);

      if (max_errs[n - 1] < sum_err){ // locate the error by changing its location one by one.
        res[n - 1] = i;
        max_errs[n - 1] = sum_err;

        int idx = n - 2;
        while (idx >= 0 && max_errs[idx] < sum_err){
          max_errs[idx + 1] = max_errs[idx];
          res[idx + 1] = res[idx];
          max_errs[idx] = sum_err;
          res[idx] = i;
          idx--;
        }
      }
    }

    return res;
  }
  #pragma endregion

  #pragma region PRINT_RESULT
  void print_results(MLP &mlp, ostream &off){ // prints results of the network (train samples first, test samples after that)
    off << "TRAIN SAMPLE RESULTS" << endl << endl;
    for(int i = 0; i < train_samples_in.size(); i++){
      vector<double> y;
      mlp.calc_s_and_o(train_samples_in[i]);
      y = mlp.get_y();
      off << train_samples_in[i] << " = " << y << " - REAL: " << train_samples_out[i] << " - ERROR: " << vabs(y - train_samples_out[i]) << endl; // {input} = {network_output} - {real_output} - {error}
    }

    off << endl << endl << "TEST SAMPLE RESULTS" << endl << endl;
    for(int i = 0; i < test_samples_in.size(); i++){
      vector<double> y;
      mlp.calc_s_and_o(test_samples_in[i]);
      y = mlp.get_y();
      off << test_samples_in[i] << " = " << y << " - REAL: " << test_samples_out[i] << " - ERROR: " << vabs(y - test_samples_out[i]) << endl; // {input} = {network_output} - {real_output} - {error}
    }
  }
  constexpr float scale_down = my_min(WIDTH, HEIGHT) / 2.0 * SAMPLE_AREA_RATIO;

#if BW == 0 && COLOUR_TYPE == 1
inline void convert_cpu(uint8_t *colors_in, uint8_t* colors_out)
{
    uint8_t region, remainder, p, q, t;
    uint8_t h = *colors_in;
    uint8_t s = *(colors_in + 1);
    uint8_t v = *(colors_in + 2);
    uint8_t *r = colors_out;
    uint8_t *g = colors_out + 1;
    uint8_t *b = colors_out + 2;

    if (s == 0)
    {
        *r = v;
        *g = v;
        *b = v;
        return;
    }

    region = h / 43;
    remainder = (h - (region * 43)) * 6;

    p = (v * (255 - s)) >> 8;
    q = (v * (255 - ((s * remainder) >> 8))) >> 8;
    t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

    switch (region)
    {
        case 0:
            *r = v; *g = t; *b = p;
            break;
        case 1:
            *r = q; *g = v; *b = p;
            break;
        case 2:
            *r = p; *g = v; *b = t;
            break;
        case 3:
            *r = p; *g = q; *b = v;
            break;
        case 4:
            *r = t; *g = p; *b = v;
            break;
        default:
            *r = v; *g = p; *b = q;
            break;
    }

    return;
}
#elif BW == 0
inline void convert_cpu(uint8_t *colors_in, uint8_t* colors_out) {
  colors_out[0] = colors_in[0];
  colors_out[1] = colors_in[1];
  colors_out[2] = colors_in[2];
}
#elif BW == 1
inline void convert_cpu(uint8_t *colors_in, uint8_t* colors_out) {
  colors_out[0] = colors_in[0];
}
#endif
  void draw_samples(QImage &im, vector<vector<double> >& in, vector<vector<double> >& out){
#if BW
    uint8_t *rowData = (uint8_t*)im.scanLine(0);
#else
    QRgb *rowData = (QRgb*)im.scanLine(0);
#endif
    for (int k = 0; k < in.size(); k++){
      int x = in[k][0] * scale_down + WIDTH / 2;
      int y = in[k][1] * scale_down + HEIGHT / 2;

      uint8_t colors_in[NETWORK_MAX_OUTPUT_SIZE] = { 0 };
      uint8_t colors_out[NETWORK_MAX_OUTPUT_SIZE] = { 0 };
      for (int j = 0; j < out[0].size(); j++){
        int res_int1 = (out[k][j] + 1.0f) * 128;
        uint8_t res_uint1 = res_int1;
        if (res_int1 > 255) res_uint1 = 255;
        if (res_int1 <   0) res_uint1 =   0;

        colors_in[j] = res_uint1;
      }

      convert_cpu(colors_in, colors_out);

      for(int i = 3; i >= -3; i--)
        for(int j = 3; j >= -3; j--)
#if BW
          *(rowData + (x + j) + (y + i) * WIDTH) = colors_out[0];
#else
          *(rowData + (x + j) + (y + i) * WIDTH) = qRgb(colors_out[0], colors_out[1], colors_out[2]);
#endif
    }
  }
  #pragma endregion

  #pragma region RANDOM
  // since Uniform_Random::generator and Uniform_Random::distribution are static, they have to be placed in one cpp file for linker to allocate space for them.
  default_random_engine generator;
  uniform_real_distribution<double> distribution;
  void init_rand(double initial_rand_max){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = default_random_engine(seed); // initialize random number generator with seed.

    distribution = uniform_real_distribution<double>(-initial_rand_max, initial_rand_max);
  }
  double get_rand(){
    return distribution(generator); // return random value
  }
  inline int get_rand_seed(){
    int seed = get_rand() * 10000.0;
    // cout << "seed: " << seed << endl;
    return seed;
  }
  float get_smooth_noise(int i, float y){
#if NOISE_TYPE == 1
    static vector<FastNoise> noisers;

    while (noisers.size() <= i){
      noisers.push_back(FastNoise(get_rand_seed()));
#if NOISE_MODE == 1
      noisers[noisers.size() - 1].SetNoiseType(FastNoise::Simplex);
#elif NOISE_MODE == 2
      noisers[noisers.size() - 1].SetNoiseType(FastNoise::Perlin);
#elif NOISE_MODE == 3
      noisers[noisers.size() - 1].SetNoiseType(FastNoise::Value);
#elif NOISE_MODE == 4
      noisers[noisers.size() - 1].SetNoiseType(FastNoise::Cellular);
#elif NOISE_MODE == 5
      noisers[noisers.size() - 1].SetNoiseType(FastNoise::Cubic);
#endif
    }

    return noisers[i].GetNoise(0.5, y);
#elif NOISE_TYPE == 2
    static vector<PerlinNoise> noisers;

    while (noisers.size() <= i){
        noisers.push_back(PerlinNoise(get_rand_seed()));
    }

    return noisers[i].noise(0.5, y * 0.001, 0.5) * 2.0 - 1.0;
#endif
  }
  QImage input_image;
  void set_image(QImage &in_img){
    input_image = QImage(in_img);
  }
  void get_from_image(double x, double y, double* r, double* g, double* b){
    int x_ = (x + 1.0) * scale_down;
    int y_ = (y + 1.0) * scale_down;

    if (x_ >= 2 * scale_down) x_ = 2 * scale_down - 1;
    if (y_ >= 2 * scale_down) y_ = 2 * scale_down - 1;

    QRgb rgb = input_image.pixel(x_, y_);
    int r_ = qRed(rgb);
    int g_ = qGreen(rgb);
    int b_ = qBlue(rgb);

    *r = r_ / 128.0 - 1.0;
    *g = g_ / 128.0 - 1.0;
    *b = b_ / 128.0 - 1.0;
  }
  void get_from_image_bw(double x, double y, double *bw){
    int x_ = (x + 1.0) * scale_down;
    int y_ = (y + 1.0) * scale_down;

    QRgb rgb = input_image.pixel(x_, y_);
    int r = qRed(rgb);
    int g = qGreen(rgb);
    int b = qBlue(rgb);

    *bw = (r + g + b) / 3.0 / 128.0 - 1.0;
  }
  #pragma endregion

  void add_train_samples(vector<vector<double> >& in, vector<vector<double> >& out){
    for (int i = 0; i < in.size(); i++){
      train_samples_in.push_back(in[i]);
      train_samples_out.push_back(out[i]);
    }
  }
  void add_test_samples(vector<vector<double> >& in, vector<vector<double> >& out){
    for (int i = 0; i < in.size(); i++){
      test_samples_in.push_back(in[i]);
      test_samples_out.push_back(out[i]);
    }
  }
  void clear_train_samples(){
    train_samples_in.clear();
    train_samples_out.clear();
  }
  void clear_test_samples(){
    test_samples_in.clear();
    test_samples_out.clear();
  }
  void enable_momentum(double coeff){
    momentum_coeff = coeff;
    use_momentum = 1;
  }
  void disable_momentum(){
    use_momentum = 0;
  }
  void set_nonlinear_output(bool type){
    use_nonlinear_output = type ? 1 : 0;
  }
  void enable_learning_rate_update(double increase_coeff, double decrease_coeff, int count){
    use_learning_rate_update = 1;
    learning_rate_update_count = count;
    learning_rate_update_decrease_coeff = decrease_coeff;
    learning_rate_update_increase_coeff = increase_coeff;
  }
  void disable_learning_rate_update(){
    use_learning_rate_update = 0;
  }
  void set_print_err_test_at_every(int n){
    print_err_test_at_every = n;
  }
  MLP getMLP(vector<int> layer_sizes, double learning_rate_, double err_threshold_, bool use_bias_neurons_){
    learning_rate = learning_rate_;
    err_threshold = err_threshold_;
    use_bias_neurons = use_bias_neurons_ ? 1 : 0;
    err_threshold_vector = vector<double>();
    for (int i = 0; i < layer_sizes[layer_sizes.size() - 1]; i++) err_threshold_vector.push_back(err_threshold);
    return MLP(layer_sizes);
  }
}
