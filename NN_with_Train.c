#include <stdint.h>
#include <math.h>
#define MAX_INPUTS 3
#define MAX_NEURONS 3
#define MAX_LAYERS 3

void Init_Neural_Network(void);
void Forward_Pass(void);
void Train(float input[], float target_val);
void Input_Acquire(void);
void Training(void);
void Test(void);

// math functions
float sigmoid(float z);
float exp_neg_tmu(float x);
float sigmoid_der(float y_hat);

//neural archtec
#define N_INPUTS 1 //inputs that is x0, x1, x2 etc; here its 1 as the its sin(x)
//define here the number of neurons in each layer
#define N_LAYERS 3
int neurons_per_layer[N_LAYERS] = {2,1,1}; //this corresponds the N_layers
int apply_output_activation = 0; //activation function at the output neurons 1-YES, 0-NO
//structures neurons

struct Neuron
{
    int n_inputs;

    float w[MAX_INPUTS];
    float b;

    float sum;   // weighted sum
    float act;   // activation output
};

struct Layer
{
    int n_neurons;
    struct Neuron neurons[MAX_NEURONS];
};

//variables

int i;
int j;
int k;

float input_layer[MAX_INPUTS];

struct Layer layers[MAX_LAYERS];
//training variables
float learning_rate=0.1f;
int e;
int epochs = 20;
float delta[MAX_LAYERS][MAX_NEURONS];
int l, n, prev;
float y_hat,sum_delta,input_val;
//test data

#define N_SAMPLES 16
//float X[N_SAMPLES][N_INPUTS];
//float Y[N_SAMPLES];
float out[N_SAMPLES];

float X[N_SAMPLES][1] =
{
    {0.000000},{0.066667},{0.133333},{0.200000},{0.266667},{0.333333},{0.400000},{0.466667},{0.533333},{0.600000},{0.666667},{0.733333},{0.800000},{0.866667},{0.933333},{1.000000}
};

float Y[N_SAMPLES] =
{
    0.000000,0.066667,0.133333,0.200000,0.266667,0.333333,0.400000,0.466667,0.533333,0.600000,0.666667,0.733333,0.800000,0.866667,0.933333,1.000000
};


int main()
{

    Init_Neural_Network();
    Training();
    Test();

    return 0;
}

void Init_Neural_Network(void)
{

for(i = 0; i < N_LAYERS; i++)
{

    layers[i].n_neurons = neurons_per_layer[i];

    for(j = 0; j < layers[i].n_neurons; j++)
    {

        if(i == 0) //check for the input layer
            layers[i].neurons[j].n_inputs = N_INPUTS; //inputs from camera picels or something like that, so n_inputs to each layer is going to be equal to the number of inputs
        else
            layers[i].neurons[j].n_inputs = neurons_per_layer[i-1]; //inputs from the previous neuron activation go it, so i-1


        /* Initialize weights */

        for(k = 0; k < layers[i].neurons[j].n_inputs; k++)
            layers[i].neurons[j].w[k] = 0.1; // initialize weights as 1 fortest

        /* Initialize outputs */

        layers[i].neurons[j].sum = 0;
        layers[i].neurons[j].act = 0;

        /* Initialize bias */

        layers[i].neurons[j].b = 0.1; //initialize as 1 for test

    }
}

}

void Forward_Pass(void)
{
    for(i = 0; i < N_LAYERS; i++)
       {
           for(j = 0; j < layers[i].n_neurons; j++)
           {
               // Start with bias
               layers[i].neurons[j].sum = layers[i].neurons[j].b;

               // Compute weighted sum
               for(k = 0; k < layers[i].neurons[j].n_inputs; k++)
               {
                   if(i == 0)
                   {
                       // First layer uses network inputs
                       layers[i].neurons[j].sum = layers[i].neurons[j].sum + layers[i].neurons[j].w[k] * input_layer[k];
                   }
                   else
                   {
                       // Other layers use previous layer activations
                       layers[i].neurons[j].sum = layers[i].neurons[j].sum + layers[i].neurons[j].w[k] * layers[i-1].neurons[k].act;
                   }
               }

               // Apply activation function
               if(i == N_LAYERS-1 && !apply_output_activation)
                   layers[i].neurons[j].act = layers[i].neurons[j].sum;  // linear output
               else
                   layers[i].neurons[j].act = sigmoid(layers[i].neurons[j].sum);
           }
       }
}


void Training(void)
{
    //TRAIN
    for(e = 0; e < epochs; e++)
    {
        for(i = 0; i < N_SAMPLES; i++)
        {
            Train(X[i], Y[i]);
        }
    }
}

void Train(float input[], float target_val)
{

    // load input
    for(k = 0; k < N_INPUTS; k++)
        input_layer[k] = input[k];

    // forward pass
    Forward_Pass();

    // delta for output layer
    int out_layer = N_LAYERS-1;
    for(j = 0; j < layers[out_layer].n_neurons; j++)
    {
           y_hat = layers[out_layer].neurons[j].act;
        delta[out_layer][j] = (y_hat - target_val) * sigmoid_der(y_hat);
    }

    // delta for hidden layers (backwards)
    for(l = N_LAYERS-2; l >= 0; l--)
    {
        for(j = 0; j < layers[l].n_neurons; j++)
        {
                sum_delta = 0.0f;
            for(k = 0; k < layers[l+1].n_neurons; k++)
            {
                sum_delta += layers[l+1].neurons[k].w[j] * delta[l+1][k];
            }
            delta[l][j] = sum_delta * sigmoid_der(layers[l].neurons[j].act);
        }
    }

    // update weights and biases
    for(l = 0; l < N_LAYERS; l++)
    {
        for(j = 0; j < layers[l].n_neurons; j++)
        {
            for(k = 0; k < layers[l].neurons[j].n_inputs; k++)
            {

                if(l == 0)
                    input_val = input_layer[k];
                else
                    input_val = layers[l-1].neurons[k].act;

                layers[l].neurons[j].w[k] = layers[l].neurons[j].w[k] - learning_rate * delta[l][j] * input_val;
            }
            layers[l].neurons[j].b = layers[l].neurons[j].b -  learning_rate * delta[l][j];
        }
    }
}




void Test(void)
{

    for(i = 0; i < N_SAMPLES; i++)
    {
        input_layer[0] = X[i][0];
        Forward_Pass();
        out[i] = layers[N_LAYERS-1].neurons[0].act;
        // send output via UART or debugger for MCU
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
