#ifndef _TRAINING_H_
#define _TRAINING_H_
#include "cell.h"
#define SAMPLE_CNT 12
#define INPUT_CNT 2
#define LEARN_RATE 0.1 // 学习速率
#define EPOCHS 1000 // 训练次数
/*
typedef struct sample {
    input_t input;
    double value;
} sample_t;
*/
typedef struct sample {
    double input[INPUT_CNT];
    double value;
} sample_t;

extern sample_t sample_set[];
extern double output_set[];

double mse_loss(void);
void train_net(void);
void train_mnet(void);
#endif