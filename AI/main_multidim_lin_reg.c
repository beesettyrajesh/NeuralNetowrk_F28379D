#define N 5
#define F 3

void lin_reg_multi_var_train();

float X[N][F] = {{1,2,1},{2,0,3},{0,1,2},{3,1,0},{2,2,1}};
float Y[N] =    {10, 13, 8, 12, 14};
float Y_pred=0;


float lr = 0.001;
float b=0;
float w[F]={0};
int i,epochs = 10000;
int j,k;

int main()
{
    lin_reg_multi_var_train();

}

void lin_reg_multi_var_train(void)
{
    float grad_w[F] = {0};
    float grad_b = 0;
    float error=0;
    for(i=0;i<epochs;i++)
    {
        //reset the gradient and bias for every epoch
        for(k=0; k<F; k++)
        {
           grad_w[k] = 0;
           grad_b = 0;
        }

        for(j=0;j<N;j++)
        {
            Y_pred = b;
            for(k=0;k<F;k++)
            {
                Y_pred = Y_pred + w[k]*X[j][k];
            }

            error = Y_pred - Y[j];
            //gradients partial derivate wrt each weight and bias
            for(k=0;k<F;k++)
            {
                grad_w[k] = grad_w[k] + error*X[j][k];
            }
            grad_b  = grad_b + error;
        }


    for(k=0;k<F;k++)
    {
        grad_w[k] = 2*grad_w[k]/N;
    }
    grad_b    = 2*grad_b/N;

    for(k=0;k<F;k++)
    {
        w[k] = w[k] - lr*grad_w[k];
    }

       b = b - lr*grad_b;
   }

}
