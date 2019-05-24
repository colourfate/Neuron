#ifndef _CELL_H_
#define _CELL_H_
#include <math.h>

#define SIGMOID(x) (1/(1+exp(-x)))
#define DSIGMOID(x) (SIGMOID(x) * (1 - SIGMOID(x)))

// 任意多个输入的神经元
typedef struct neuron {
    int inum;
    double b;
    double *w;
} neuron_t;

// 隐藏层任意个神经元，输出层一个神经元
typedef struct neuron_net {
    int hnum;
    neuron_t *h;
    neuron_t o;
} neuron_net_t;

// 包含两个输入的神经元
typedef struct cell {
    double w1, w2, b;
} cell_t;

typedef struct input {
    double x1, x2;
} input_t;

// 隐藏层两个神经元，输出层一个神经元
typedef struct cell_net {
    cell_t c1, c2, o1;
} cell_net_t;

extern cell_net_t net;
double feedforward(cell_t *cell, input_t *input);
double dfeedforward(cell_t *cell, input_t *input);
//double net_feedforward(input_t *input);

extern neuron_net_t mnet;
int neuron_init(neuron_t *n, int input_num);
double neuron_feedforward(neuron_t *n, double *input, int deriv);
void neuron_release(neuron_t *n);
int net_init(int neuron_num, int input_num);
void net_release(void);
void net_show(void);
int net_save(void);
int net_load(void);
double net_feedforward(double *input);
#endif