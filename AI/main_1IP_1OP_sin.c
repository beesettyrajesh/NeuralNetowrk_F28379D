#define N 12      // number of samples

void sin_per_two_hid_train();
float sigmoid(float z);
float exp_neg_tmu(float x);
float sigmoid_der(float y_hat);

float X[N] = {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5};
float Y[N] = {0, 0.479, 0.841, 0.997, 0.909, 0.598, 0.141, -0.351, -0.757, -0.977, -0.959, -0.705};

int epochs=10000;
float lr=0.01;

float w1=0.1f,w2=0.1f;
float b1,b2;
float y1,y2;
float v1=0.1f,v2=-.1f;
float bo;
float h1,h2;
float Yo;
float Yt[12];

int i,j;

int main()
{
    sin_per_two_hid_train();

    for(j=0;j<N;j++)
        {
            float h1 = sigmoid(w1*X[j] + b1);
            float h2 = sigmoid(w2*X[j] + b2);
            Yt[j] = v1*h1 + v2*h2 + bo;
        }
}

void sin_per_two_hid_train(void)
{
    float error,del0,del1,del2;

    for(i=0;i<epochs;i++)
    {
        for(j=0;j<N;j++)
        {
            h1 = sigmoid(w1*X[j] + b1);
            h2 = sigmoid(w2*X[j] + b2);
            Yo = h1*v1 + h2*v2 + bo;

            error = Yo - Y[j];
            del0 = error;

            del1 = v1*del0*sigmoid_der(h1); //chain rule partial derivat
            del2 = v2*del0*sigmoid_der(h2);

            v1   = v1 - lr*del0*h1;  //update weights
            v2   = v2 - lr*del0*h2;
            bo   = bo - lr*del0;

            w1   = w1 - lr*del1*X[j];
            w2   = w2 - lr*del2*X[j];
            b1   = b1 - lr*del1;
            b2   = b2 - lr*del2;
        }
    }
}

float sigmoid(float z)
{
    return 1.0f / (1.0f + exp_neg_tmu(z));
}

float exp_neg_tmu(float x)
{
    const float LOG2E = 1.44269504089f;
    return __iexp2(-x * LOG2E); // MCU intrinsic
}

float sigmoid_der(float y_hat)
{
    return y_hat * (1.0f - y_hat);
}
