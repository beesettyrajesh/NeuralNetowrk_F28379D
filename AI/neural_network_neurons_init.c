#include <stdint.h>

#define MAX_INPUTS 5
#define MAX_NEURONS 10
#define MAX_LAYERS 5

void Init_Neural_Network(void);
void Fowrward_Pass(void);

// math functions
float sigmoid(float z);
float exp_neg_tmu(float x);
float sigmoid_der(float y_hat);

//neural archtec
#define N_INPUTS 3
#define N_LAYERS 3

//define here the number of neurons in each layer
int neurons_per_layer[N_LAYERS] = {2,5,1};
int apply_output_activation = 1; //activation function at the output neurons 1-YES, 0-NO
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


int main()
{
    input_layer[0] = 1.0f;
    input_layer[1] = 2.0f;
    input_layer[2] = 3.0f;

    Init_Neural_Network();

    Fowrward_Pass();

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
            layers[i].neurons[j].w[k] = 1; // initialize weights as 1 fortest

        /* Initialize outputs (NO computation) */

        layers[i].neurons[j].sum = 0;
        layers[i].neurons[j].act = 0;

        /* Initialize bias */

        layers[i].neurons[j].b = 1; //initialize as 1 for test

    }
}

}

void Fowrward_Pass(void)
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
