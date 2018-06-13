#ifndef DEFINITIONS
#define DEFINITIONS

constexpr int train_sample_count = 10;
constexpr int mlp_count = 4; // increase this for more complex images

#define COMBINE_MODE 5 // 1 (average) - 2 (max) - 3 (min) - 4 (multiply) - 5 (overlay) - 6 (perlinnoise - mlp_count should be 2)
static_assert(COMBINE_MODE != 6 || (COMBINE_MODE == 6 && mlp_count == 2), "ERROR - COMBINE_MODE 6 requires mlp_count=2");

#define BW 0 // 0 (COLOUR) - 1 (BW)
#if BW == 0
    #define COLOUR_TYPE 1 // 0 (RGB) - 1 (HSV)
#endif

#define RAND_MODE 3 // 1 (allRAND) - 2 (incrRAND) - 3 (smoothRand(noise))
#define FROM_IMAGE 0 // 1 (get_from_image (requires BW == 0 && COLOUR_TYPE == 0)) - 0

static_assert(FROM_IMAGE == 0 || (FROM_IMAGE == 1 && BW == 0 && COLOUR_TYPE == 0), "ERROR - FROM_IMAGE requires BW=0 and COLOUR_TYPE=0");

#if RAND_MODE == 3
    #define NOISE_TYPE 2 // 1 (FastNoise) - 2 (perlinnoise)
    #if NOISE_TYPE == 1
        #define NOISE_MODE 2 // 1 (Simplex) - 2 (Perlin) - 3 (Value) - 4 (Cellular) - 5 (Cubic)
    #endif
#endif

// possible values for act(activation function) --
// -- act_sig - act_fs - act_sin - act_sinc - act_gauss - act_relu - act_softplus
#define ACT act_fs

#define ACT1 ACT
#define ACT2 ACT
#define ACT3 ACT
#define ACT4 ACT
#define ACT5 ACT
#define ACTDEF ACT
#define ACTLAST act_fs // this one should output -1.0 to 1.0

#define WIDTH 250
#define HEIGHT 250
#define SAMPLE_AREA_RATIO 0.8 // this should be smaller than 1.0
// NOTE: if FROM_IMAGE is used, then the dimensions of the image should be bigger than or equal to
//min(WIDTH, HEIGHT) * SAMPLE_AREA_RATIO

#define USE_SMOOTHNESS_WHILE_RESCALING_FOR_SCREEN 0 // make this 1 for smoother images (reduces fps drastically in some systems)

#ifndef CUDA_FILE
    #include <QString>

    const QString in_file_name = "field.jpg"; // used only when FROM_IMAGE is 1
    const QString save_file_prefix = "image";
#endif

#define RESCALE_WHEN_SAVING 0 // this always uses smooth rescaling // this is actually useless because it can be done afterwards too
                              //for example when using ffmpeg to create video from saved images ffmpeg can rescale using very good smoothness
                              //algorithms like "lanczos"
#if RESCALE_WHEN_SAVING
    #define SAVE_WIDTH 800
    #define SAVE_HEIGHT 800
#endif


#pragma region LAYERS
#if BW
#define LAST_LAYER 1
#else
#define LAST_LAYER 3
#endif

//constexpr int layer_sizes_size = 5;
//constexpr int layer_sizes[layer_sizes_size] = {2, 6, 10, 6, LAST_LAYER};
//constexpr int layer_sizes_size = 8;
//constexpr int layer_sizes[layer_sizes_size] = {2, 8, 18, 30, 20, 15, 6, LAST_LAYER};
constexpr int layer_sizes_size = 4;
constexpr int layer_sizes[layer_sizes_size] = {2, 4, 4, LAST_LAYER};
//constexpr int layer_sizes_size = 8;
//constexpr int layer_sizes[layer_sizes_size] = {2, 20, 48, 60, 45, 25, 12, LAST_LAYER};
//constexpr int layer_sizes_size = 4;
//constexpr int layer_sizes[layer_sizes_size] = {2, 50, 50, LAST_LAYER};

extern int layer_starts_w[layer_sizes_size - 1];
void fill_layer_starts_w();
#pragma endregion



//#define STRANGE_VERSION // if you open this, it will produce strange images but there is a bug in this version so I'm not sure how it works
#define USE_BIAS 1 // 0 - 1
#define LEARNING_RATE_UPDATE 1 // 0 - 1

#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__
#define CONCAT_T(act) PRIMITIVE_CAT(act, t)
#define ACT1t CONCAT_T(ACT1)
#define ACT2t CONCAT_T(ACT2)
#define ACT3t CONCAT_T(ACT3)
#define ACT4t CONCAT_T(ACT4)
#define ACT5t CONCAT_T(ACT5)
#define ACTDEFt CONCAT_T(ACTDEF)
#define ACTLASTt CONCAT_T(ACTLAST)

#endif // DEFINITIONS

