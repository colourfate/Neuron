#include <stdlib.h>
#include <stdio.h>
#include "cell.h"

cell_net_t net = {
    .c1 = {0, 1, 0},
    .c2 = {0, 1, 0},
    .o1 = {0, 1, 0}
};

double feedforward(cell_t *cell, input_t *input)
{
    double tatol = input->x1 * cell->w1 + input->x2 * cell->w2 + cell->b;
    return SIGMOID(tatol);
}

double dfeedforward(cell_t *cell, input_t *input)
{
    double tatol = input->x1 * cell->w1 + input->x2 * cell->w2 + cell->b;
    return DSIGMOID(tatol);
}

/*
double net_feedforward(input_t *input)
{
    input_t temp;

    temp.x1 = feedforward(&net.c1, input);
    temp.x2 = feedforward(&net.c2, input);
    
    return feedforward(&net.o1, &temp);
}
*/

/**************************************************/

int neuron_init(neuron_t *n, int input_num)
{   
    int i;

    if(input_num > 64 || input_num < 0)
        return -1;
    
    n->inum = input_num;
    n->b = 0;
    n->w = malloc(sizeof(double) * input_num);
    n->w[0] = 1;
    for(i = 1; i < input_num; i++)
        n->w[i] = 0;
}

void neuron_release(neuron_t *n)
{
    free(n->w);
    n->inum = 0;
}

double neuron_feedforward(neuron_t *n, double *input, int deriv)
{
    double tatol = 0;
    int i;

    for(i = 0; i < n->inum; i++)
        tatol += input[i] * n->w[i];
    tatol += n->b;

    if(deriv == 0)
      return SIGMOID(tatol);
    else if(deriv == 1)
      return DSIGMOID(tatol);
}

neuron_net_t mnet;

int net_init(int neuron_num, int input_num)
{
    int i;

    if(neuron_num > 64 || neuron_num < 0)
        return -1;

    mnet.hnum = neuron_num;

    mnet.h = malloc(sizeof(neuron_t) * neuron_num);
    for(i = 0; i < neuron_num; i++)
      neuron_init(&mnet.h[i], input_num);

    neuron_init(&mnet.o, neuron_num);
}

void net_release(void)
{
    int i;

    neuron_release(&mnet.o);
    for(i = 0; i < mnet.hnum; i++)
      neuron_release(&mnet.h[i]);
    
    free(mnet.h);
    mnet.h = NULL;
    mnet.hnum = 0;
}

void net_show(void)
{
    int i, j;

    printf("hidden layer num: %d\n", mnet.hnum);
    for(i = 0; i < mnet.hnum; i++) {
        printf("[h%d]: %f", i, mnet.h[i].b);
        for(j = 0; j < mnet.h[i].inum; j++)
            printf(", %f", mnet.h[i].w[j]);
        printf("\n");
    }
    printf("[o]: %f", mnet.o.b);
    for(j = 0; j < mnet.o.inum; j++)
        printf(", %f", mnet.o.w[j]);
    printf("\n");
}

int net_save(void)
{
    FILE *fp;
    int i, j;
    
    fp = fopen("param", "w");
    if(!fp)
      return -1;

    fprintf(fp, "%d %d\n", mnet.hnum, mnet.h[0].inum);
    for(i = 0; i < mnet.hnum; i++) {
        fprintf(fp, "%.9f", mnet.h[i].b);
        for(j = 0; j < mnet.h[i].inum; j++)
            fprintf(fp, ",%.9f", mnet.h[i].w[j]);
        fprintf(fp, "\n");
    }
    fprintf(fp, "%.9f", mnet.o.b);
    for(j = 0; j < mnet.o.inum; j++)
        fprintf(fp, ",%.9f", mnet.o.w[j]);
    fprintf(fp, "\n");

    fclose(fp);

    return 0;
}

int net_load(void)
{
    FILE *fp;
    int i, j;
    int hnum, inum;
    
    fp = fopen("param", "r");
    if(!fp)
      return -1;

    fscanf(fp, "%d %d", &hnum, &inum);
    net_init(hnum, inum);
    for(i = 0; i < hnum; i++) {
        fscanf(fp, "%lf", &mnet.h[i].b);
        for(j = 0; j < inum; j++) {
            fscanf(fp, ",%lf", &mnet.h[i].w[j]);
        }
    }
    fscanf(fp, "%lf", &mnet.o.b);
    for(j = 0; j < inum; j++) {
        fscanf(fp, ",%lf", &mnet.o.w[j]);
    }

    printf("load net\n");
    net_show();

    fclose(fp);

    return 0;
}

double net_feedforward(double *input)
{
    int hide_num = mnet.hnum;
    double *h = malloc(sizeof(double) * hide_num);
    double res = 0;
    int i;

    for(i = 0; i < hide_num; i++)
        h[i] = neuron_feedforward(&mnet.h[i], input, 0);
    res = neuron_feedforward(&mnet.o, h, 0);

    free(h);
    return res;
}