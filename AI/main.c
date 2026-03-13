#define N 5

int lin_reg_train();
// Training data
float X[N] = {1, 2, 3, 4, 5};
float Y[N] = {2, 4, 6, 8, 10};  // Example: y = 2*x

float w=0,b=0;

float lr = 0.01;
int epochs = 1000;
int i,j;



int main()
{
    lin_reg_train();

}

lin_reg_train()
{
    float error = 0;
    float grad_wrt_w = 0;
    float grad_wrt_b = 0;

    for(i=0;i<epochs;i++)
    {
        grad_wrt_w = 0;
        grad_wrt_b = 0;

        for(j=0;j<N;j++)
        {
            error = (w*X[j] + b) - Y[j];

            grad_wrt_w = grad_wrt_w + error * X[j];
            grad_wrt_b = grad_wrt_b + error;
        }

        grad_wrt_w = (2*grad_wrt_w)/N;
        grad_wrt_b = (2*grad_wrt_b)/N;

        w = w - lr*grad_wrt_w;
        b = b - lr*grad_wrt_b;
    }

}
