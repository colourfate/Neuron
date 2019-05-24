#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cell.h"
#include "training.h"

int main(int argc, char *argv[])
{
    if(argc != 3) {
        printf("./net [height] [weight]\n");
        return -1;
    }

    net_init(2, INPUT_CNT);
    net_show();
    train_mnet();
    net_save();

    net_load();
    double input[2];
    input[0] = atoi(argv[1]) - 160;
    input[1] = atoi(argv[2]) - 50;
    printf("[%s %s]: %f\n", argv[1], argv[2], net_feedforward(input));
    net_release();

    return 0;
}