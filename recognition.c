#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>
#include <arm_neon.h>
#include <pthread.h>

#define SIGMOID_TABLE_OFFSET 1000
#define SIGMOID_TABLE_FACTOR 10.0
#define sigmoid(x) (sigmoid_table[(int)((x) * SIGMOID_TABLE_FACTOR) + SIGMOID_TABLE_OFFSET])
#define SIZE1 size*IMG_SIZE
#define SIZE2 size*size
#define NUM_THREADS 4

float* hidden_layers1, * hidden_layers2, ** biases, * temp, ** weights;
int* labels;
float* confidences;
float sigmoid_table[2001];

float array1[NUM_THREADS][sizeof(float) * 512];
float array2[NUM_THREADS][sizeof(float) * 512];

void init_sigmoid_table()
{
    int i;
    for (i = -1000; i <= 1000; i++)
    {
        float x = (float)i / 10;

        sigmoid_table[i + SIGMOID_TABLE_OFFSET] = 1 / (1 + exp(-x));
    }
}

typedef struct {
    float* hidden_layers1;
    float* hidden_layers2;
    float** weights;
    float** biases;
    int size;
    int depth;
    float* images;
    int* labels;
    float* confidences;
    int start_idx;
    int end_idx;
    long tid;
} ThreadData;

void* recognize_numbers_thread(void* arg) {

    ThreadData* data = (ThreadData*)arg;
    int size = data->size;
    int depth = data->depth;
    float* images = data->images;
    labels = data->labels;
    confidences = data->confidences;

    int tid = data->tid;
    int images_per_thread = IMG_COUNT / NUM_THREADS;
    int start_idx = tid * (images_per_thread);
    int end_idx = start_idx + (images_per_thread);

    if(tid == 3){
        end_idx += (IMG_COUNT % NUM_THREADS);
    }


    int i, j, x, y;
    for (i = start_idx; i < end_idx; i += 2) {
        float* input1 = images + IMG_SIZE * i;
        float* input2 = images + IMG_SIZE * (i + 1);
        float output1[NUM_THREADS][DIGIT_COUNT];
        float output2[NUM_THREADS][DIGIT_COUNT];

        // From the input layer to the first hidden layer
        for (x = 0; x < size; x += 2)
        {
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;
            float sum4 = 0.0f;
            // initialize
            float32x4_t sum1_vec = vdupq_n_f32(0.0f);
            float32x4_t sum2_vec = vdupq_n_f32(0.0f);
            float32x4_t sum3_vec = vdupq_n_f32(0.0f);
            float32x4_t sum4_vec = vdupq_n_f32(0.0f);

            int size3 = IMG_SIZE * x;
            int size4 = IMG_SIZE * (x + 1);
            for (y = 0; y < IMG_SIZE; y += 4)
            {
                float32x4_t input1_vec = vld1q_f32(input1 + y);
                float32x4_t input2_vec = vld1q_f32(input2 + y);
                float32x4_t weights_vec1 = vld1q_f32(weights[0] + size3 + y);
                float32x4_t weights_vec2 = vld1q_f32(weights[0] + size4 + y);

                sum1_vec = vmlaq_f32(sum1_vec, input1_vec, weights_vec1);
                sum2_vec = vmlaq_f32(sum2_vec, input1_vec, weights_vec2);
                sum3_vec = vmlaq_f32(sum3_vec, input2_vec, weights_vec1);
                sum4_vec = vmlaq_f32(sum4_vec, input2_vec, weights_vec2);
            }

            sum1 += (sum1_vec[0] + sum1_vec[1] + sum1_vec[2] + sum1_vec[3] + biases[0][x]);
            sum2 += (sum2_vec[0] + sum2_vec[1] + sum2_vec[2] + sum2_vec[3] + biases[0][x + 1]);
            sum3 += (sum3_vec[0] + sum3_vec[1] + sum3_vec[2] + sum3_vec[3] + biases[0][x]);
            sum4 += (sum4_vec[0] + sum4_vec[1] + sum4_vec[2] + sum4_vec[3] + biases[0][x + 1]);

            array1[tid][x] = sigmoid(sum1);
            array1[tid][x + 1] = sigmoid(sum2);
            array2[tid][x] = sigmoid(sum3);
            array2[tid][x + 1] = sigmoid(sum4);
        }

        // Between hidden layers
        for (j = 1; j < depth; j++)
        {
            for (x = 0; x < size; x++)
            {
                float sum1 = 0;
                float sum2 = 0;
                int size3 = size * (j - 1);
                int size4 = size * x;
                for (y = 0; y < size; y++)
                {
                    sum1 += array1[tid][size3 + y] * weights[j][size4 + y];
                    sum2 += array2[tid][size3 + y] * weights[j][size4 + y];
                }
                sum1 += biases[j][x];
                sum2 += biases[j][x];

                array1[tid][size * j + x] = sigmoid(sum1);
                array2[tid][size * j + x] = sigmoid(sum2);
            }
        }

        // From the last hidden layer to the output layer
        for (x = 0; x < DIGIT_COUNT; x++)
        {
            float sum1 = 0;
            float sum2 = 0;
            int size3 = size * (depth - 1);
            int size4 = size * x;
            for (y = 0; y < size; y++)
            {
                sum1 += array1[tid][size3 + y] * weights[depth][size4 + y];
                sum2 += array2[tid][size3 + y] * weights[depth][size4 + y];
            }
            sum1 += biases[depth][x];
            sum2 += biases[depth][x];

            output1[tid][x] = sigmoid(sum1);
            output2[tid][x] = sigmoid(sum2);

        }

        // Find the answer
        float max1 = 0, max2 = 0;
        int label1 = 0, label2 = 0;
        for (x = 0; x < DIGIT_COUNT; x++)
        {
            if (output1[tid][x] > max1)
            {
                label1 = x;
                max1 = output1[tid][x];
            }
            if (output2[tid][x] > max2)
            {
                label2 = x;
                max2 = output2[tid][x];
            }
        }

        // Store the result
        confidences[i] = max1;
        labels[i] = label1;
        confidences[i + 1] = max2;
        labels[i + 1] = label2;
    }
    return NULL;
}



void Recognize_numbers(float* hidden_layers1, float* hidden_layers2, float** weights, float** biases, int size, int depth, float* images, int* labels, float* confidences) {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    int start_idx = 0;
    int end_idx = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].hidden_layers1 = hidden_layers1;
        thread_data[i].hidden_layers2 = hidden_layers2;
        thread_data[i].weights = weights;
        thread_data[i].biases = biases;
        thread_data[i].size = size;
        thread_data[i].depth = depth;
        thread_data[i].images = images;
        thread_data[i].labels = labels;
        thread_data[i].confidences = confidences;
        thread_data[i].start_idx = start_idx;
        thread_data[i].end_idx = end_idx;
        thread_data[i].tid = i;

        pthread_create(&threads[i], NULL, recognize_numbers_thread, (void*)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}


void Output_layer(float** weights, float** biases, int size, int depth) {
    weights[depth] = weights[depth - 1] + SIZE2 + size;
    biases[depth] = weights[depth] + DIGIT_COUNT * size;

    return;
}

void Hidden_layers(float** weights, float** biases, float* network, int size, int depth) {
    int i;

    for (i = 1; i < depth; i++)
    {
        weights[i] = network + (SIZE1 + size) + (SIZE2 + size) * (i - 1);
        biases[i] = weights[i] + SIZE2;
    }

    return;
}

void Input_layer(float** weights, float** biases, float* network, int size) {
    weights[0] = network;
    biases[0] = network + SIZE1;

    return;
}

void recognition(float* images, float* network, int depth, int size, int* labels, float* confidences)
{
    int flo = sizeof(float);

    // Aligned memory allocation
    if (posix_memalign((void**)&hidden_layers1, 64, flo * size) != 0) {
        // Handle memory allocation error
        printf("Memory allocation error for hidden_layers\n");
        exit(1);
    }

    if (posix_memalign((void**)&hidden_layers2, 64, flo * size) != 0) {
        // Handle memory allocation error
        printf("Memory allocation error for hidden_layers\n");
        exit(1);
    }

    if (posix_memalign((void**)&weights, 64, flo * (depth + 1)) != 0) {
        // Handle memory allocation error
        printf("Memory allocation error for weights\n");
        free(hidden_layers1);
        free(hidden_layers2);
        exit(1);
    }

    if (posix_memalign((void**)&biases, 64, flo * (depth + 1)) != 0) {
        // Handle memory allocation error
        printf("Memory allocation error for biases\n");
        free(hidden_layers1);
        free(hidden_layers2);
        free(weights);
        exit(1);
    }


    // Set pointers for weights and biases
    // 1. Input layer
    Input_layer(weights, biases, network, size);
    // 2. Hidden layers
    Hidden_layers(weights, biases, network, size, depth);
    // 3. Output layer
    Output_layer(weights, biases, size, depth);

    // Initialize the sigmoid table
    init_sigmoid_table();

    // Recognize numbers
    Recognize_numbers(hidden_layers1, hidden_layers2, weights, biases, size, depth, images, labels, confidences);

    // Free aligned memory
    free(hidden_layers1);
    free(hidden_layers2);
    free(weights);
    free(biases);
}