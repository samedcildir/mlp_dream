#ifndef GLOBALS_HPP
#define GLOBALS_HPP

extern int save_cnt;

constexpr int my_min(int x, int y){
    return x > y ? x : y;
}

#define NETWORK_MAX_OUTPUT_SIZE 10 // this is 1 for BW, 3 for color but 10 makes it solid

#endif // GLOBALS_HPP
