#include <math.h>
#define N 5

void lin_reg_per_train();

float X[N] = {1, 2, 3, 4, 5};
float Y[N] = {0.1, 0.3, 0.5, 0.7, 0.9};  // scaled for sigmoid

float Y_act=0;
float Y_test;
float lr = 0.0001;
float b=0;
float w= 0;
int i,j,epochs = 10000;

int main()
{

    lin_reg_per_train();
    Y_test = 5*w + b;

}

void lin_reg_per_train(void)
{
    float sig_d,error = 0;
    float grad_wrt_w = 0;
    float grad_wrt_b = 0;
    float Y_pred=0;
    for(i=0;i<epochs;i++)
    {
        grad_wrt_w = 0;
        grad_wrt_b = 0;

        for(j=0;j<N;j++)
        {
            Y_pred = (w*X[j] + b);
            Y_act  = 1/(1+exp(-Y_pred)); //activation function output
            sig_d  = Y_act/(1-Y_act);  //derivative of sigmoid
            error  = Y_act - Y[j];
            //gradients
            grad_wrt_w = grad_wrt_w + error*sig_d*X[j];
            grad_wrt_b = grad_wrt_b + error*sig_d;
        }

        //average them and update

        grad_wrt_w = (2.0/N)*grad_wrt_w;
        grad_wrt_b = (2.0/N)*grad_wrt_b;

        w = w - lr*grad_wrt_w;
        b = b - lr*grad_wrt_b;
    }
}
