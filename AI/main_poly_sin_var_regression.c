#include <math.h>
#define D 2 //Degree
#define N 14

void lin_reg_multi_var_train();

float X[N] =    {1, 2, 3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
float Y[N] =    {1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196};
float Y_pred=0;

float lr = 0.0000001;
float b=0;
float w[D]={0};
int i,epochs = 10000;
int j,k;

int main()
{
    lin_reg_multi_var_train();

}

void lin_reg_multi_var_train()
{
    float grad_w[D] = {0};
    float grad_b = 0;
    float error=0;

    for(i=0;i<epochs;i++)
    {
        for(k=0;k<D;k++)
         {
             grad_w[k] = 0;
         }
         grad_b = 0;

       for(j=0;j<N;j++)
        {
            Y_pred = b;

            for(k=0;k<D;k++)
            {
                Y_pred =  Y_pred + w[k]*pow(X[j],k+1);
            }

            error = Y_pred-Y[j];

            for(k=0;k<D;k++)
            {
                grad_w[k] = grad_w[k] + (error*pow(X[j],k+1));
            }

            grad_b = grad_b + error;


        }

       for(k=0;k<D;k++)
       {
           grad_w[k] = (2.0/N)*(grad_w[k]);
       }
       grad_b = (2.0/N)*(grad_b);


        for(k=0;k<D;k++)
        {
            w[k] = w[k] - lr*grad_w[k];
        }
            b = b - lr*grad_b;
    }


}
