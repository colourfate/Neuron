#include <stdio.h>
#include <stdlib.h>
#include "training.h"

// 输入身高（135cm为标准），体重（66公斤为标准），输出性别（1为男，0为女）
/*
sample_t sample_set[SAMPLE_CNT] = {
    {{-2, -1}, 1},
    {{25, 6}, 0},
    {{17, 4}, 0},
    {{-15, -6}, 1}
};
*/
// 身高以160为标准，体重以50kg为标准
sample_t sample_set[SAMPLE_CNT] = {
    {{3, 10}, 1},
    {{-7, -8}, 0},
    {{10, 20}, 1},
    {{-1, -7}, 0},
    {{12, 10}, 1},
    {{1, 0}, 0},
    {{15, 20}, 1},
    {{3, -4}, 0},
    {{15, 12}, 1},
    {{4, -3}, 0},
    {{20, 21}, 1},
    {{10, 4}, 0}
};

double output_set[SAMPLE_CNT];

double mse_loss(void)
{
    int i;
    double err, sum = 0;

    for(i = 0; i < SAMPLE_CNT; i++) {
        err = sample_set[i].value - output_set[i];
        sum += err * err;
    }

    return sum / SAMPLE_CNT;
}

/*
void train_net(void)
{
    int i, j;
    double pred, value;    // output predicition和output ture
    input_t mid;
    double dL_dpred;
    double dpred_do1w1, dpred_do1w2, dpred_do1b, dpred_do1x1, dpred_do1x2;
    double do1x1_dc1w1, do1x1_dc1w2, do1x1_dc1b;
    double do1x2_dc2w1, do1x2_dc2w2, do1x2_dc2b;

    printf("mse loss: \n");
    for(j = 0; j < EPOCHS; j++) {
        for(i = 0; i < SAMPLE_CNT; i++) {
            mid.x1 = feedforward(&net.c1, &sample_set[i].input);
            mid.x2 = feedforward(&net.c2, &sample_set[i].input);
            pred = feedforward(&net.o1, &mid);
            value = sample_set[i].value;

            // L为方差，方差对当前输出预测的偏导，方差中其他项不包含当前输出预测，导数为0
            dL_dpred = -2 * (value - pred);

            // 神经元o1
            dpred_do1w1 = mid.x1 * dfeedforward(&net.o1, &mid);
            dpred_do1w2 = mid.x2 * dfeedforward(&net.o1, &mid);
            dpred_do1b = dfeedforward(&net.o1, &mid);

            dpred_do1x1 = net.o1.w1 * dfeedforward(&net.o1, &mid);
            dpred_do1x2 = net.o1.w2 * dfeedforward(&net.o1, &mid);
            
            // 神经元c1
            do1x1_dc1w1 = sample_set[i].input.x1 * dfeedforward(&net.c1, &sample_set[i].input);
            do1x1_dc1w2 = sample_set[i].input.x2 * dfeedforward(&net.c1, &sample_set[i].input);
            do1x1_dc1b = dfeedforward(&net.c1, &sample_set[i].input);

            // 神经元c2
            do1x2_dc2w1 = sample_set[i].input.x1 * dfeedforward(&net.c2, &sample_set[i].input);
            do1x2_dc2w2 = sample_set[i].input.x2 * dfeedforward(&net.c2, &sample_set[i].input);
            do1x2_dc2b = dfeedforward(&net.c2, &sample_set[i].input);

            // 使用SGD算法更新权值和偏置
            net.c1.w1 -= LEARN_RATE * dL_dpred * dpred_do1x1 * do1x1_dc1w1;
            net.c1.w2 -= LEARN_RATE * dL_dpred * dpred_do1x1 * do1x1_dc1w2;
            net.c1.b -= LEARN_RATE * dL_dpred * dpred_do1x1 * do1x1_dc1b;

            net.c2.w1 -= LEARN_RATE * dL_dpred * dpred_do1x2 * do1x2_dc2w1;
            net.c2.w2 -= LEARN_RATE * dL_dpred * dpred_do1x2 * do1x2_dc2w2;
            net.c2.b -= LEARN_RATE * dL_dpred * dpred_do1x2 * do1x2_dc2b;

            net.o1.w1 -= LEARN_RATE * dL_dpred * dpred_do1w1;
            net.o1.w2 -= LEARN_RATE * dL_dpred * dpred_do1w2;
            net.o1.b -= LEARN_RATE * dL_dpred * dpred_do1b;

            output_set[i] = pred;
        }
        if(j % 10 == 0)
            printf("%.9f\n", mse_loss());
    }
    printf("training complete:\n");
    printf("c1: %f, %f, %f\n", net.c1.w1, net.c1.w2, net.c1.b);
    printf("c2: %f, %f, %f\n", net.c2.w1, net.c2.w2, net.c2.b);
    printf("o1: %f, %f, %f\n", net.o1.w1, net.o1.w2, net.o1.b);
}
*/

void train_mnet(void)
{
    int i, j, k, l;
    int hide_num;
    double *h;
    double pred, value;
    double dL_dpred;

    hide_num = mnet.hnum;
    h = malloc(sizeof(double) * hide_num);

    for(i = 0; i < EPOCHS; i++) {
        for(j = 0; j < SAMPLE_CNT; j++) {
            for(k = 0; k < hide_num; k++)
                h[k] = neuron_feedforward(&mnet.h[k], sample_set[j].input, 0);
            pred = neuron_feedforward(&mnet.o, h, 0);
            value = sample_set[j].value;

            dL_dpred = -2 * (value - pred);

            // output
            double temp = neuron_feedforward(&mnet.o, h, 1);
            for(k = 0; k < hide_num; k++) {
                double dpred_dow = h[k] * temp;
                mnet.o.w[k] -= LEARN_RATE * dL_dpred * dpred_dow;
            }
            double dpred_dob = temp;
            mnet.o.b -= LEARN_RATE * dL_dpred * dpred_dob;
            /*
            for(k = 0; k < hide_num; k++)
                dpred_doi[k] = mnet.o.w[k] * temp;
            */

            // hide
            for(k = 0; k < hide_num; k++) {
                double dpred_doi = mnet.o.w[k] * temp;
                temp = neuron_feedforward(&mnet.h[k], sample_set[j].input, 1);
                for(l = 0; l < INPUT_CNT; l++) {
                    double doi_dhw = sample_set[j].input[l] * temp;
                    mnet.h[k].w[l] -= LEARN_RATE * dL_dpred * dpred_doi * doi_dhw;
                }
                double doi_dhb = temp;
                mnet.h[k].b -= LEARN_RATE * dL_dpred * dpred_doi * doi_dhb;
            }

            output_set[j] = pred;
        }
        //if(i % 10 == 0)
        //    printf("%f\n", mse_loss());
    }
    net_show();

    free(h);
}