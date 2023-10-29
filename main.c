#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "recognition.h"

int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);
void load_MNIST(float * images, int * labels);

JNIEXPORT void JNICALL
Java_com_example_jni_1sum_MainActivity_runMain(JNIEnv* env,jclass clazz, jstring path_network,jstring path_out,
                                               jcharArray buffer) {
// TODO: implement runMain()
const char *nativePath = (*env)->GetStringUTFChars(env, path_network, NULL);
jchar *str = (*env)->GetCharArrayElements(env, buffer, NULL);
if (nativePath == NULL) {
return ; // Error handling
}

// TODO: implement runMain()

float *images;
float *network;
float *confidences;
int *labels;
int *labels_ans;
int i, correct, total_network_size;
float accuracy;

FILE *io_file;
struct timespec start, end, spent;

int size = 4096;
int depth = 3;

images = (float *)malloc(sizeof(float) * IMG_COUNT * IMG_SIZE);
labels = (int *)malloc(sizeof(int) * IMG_COUNT);
labels_ans = (int *)malloc(sizeof(jint) * IMG_COUNT);
confidences = (float *)malloc(sizeof(float) * IMG_COUNT);

io_file = fopen(nativePath, "r");
if (!io_file) {
fprintf(stderr, "Invalid network file %s!\n", nativePath);
exit(EXIT_FAILURE);
}
fread(&depth, sizeof(int), 1, io_file);
fread(&size, sizeof(int), 1, io_file);
printf("size=%d, depth=%d\n", size, depth);
total_network_size = (IMG_SIZE * size + size) + (depth - 1) * (size * size + size) + size * DIGIT_COUNT + DIGIT_COUNT;
network = (float *)malloc(sizeof(float) * total_network_size);
fread(network, sizeof(float), total_network_size, io_file);
fclose(io_file);

io_file = fopen("MNIST_image.bin", "r");
fread(images, sizeof(float), IMG_COUNT * IMG_SIZE, io_file);
fclose(io_file);

io_file = fopen("MNIST_label.bin", "r");
fread(labels_ans, sizeof(int), IMG_COUNT, io_file);
fclose(io_file);

clock_gettime(CLOCK_MONOTONIC, &start);
recognition(images, network, depth, size, labels, confidences);
clock_gettime(CLOCK_MONOTONIC, &end);
timespec_subtract(&spent, &end, &start);

correct = 0;
for (i = 0; i < IMG_COUNT; i++) {
if (labels_ans[i] == labels[i])
correct++;
}

accuracy = (float)correct / (float)IMG_COUNT;
jchar tmp[50] = {0};
strcpy(str, "Elapsed time: ");
sprintf(tmp, "%ld", spent.tv_sec);
strcat(str, tmp);
strcat(str, ".");
sprintf(tmp, "%03ld", spent.tv_nsec / 1000 / 1000);
strcat(str, tmp);
strcat(str, "sec\n");
strcat(str, "Accuracy: ");
strcat(str, tmp);
sprintf(tmp, "%ld", spent.tv_nsec / 1000 / 1000);
strcat(str, "Accuracy: ");
sprintf(tmp, "%.3f", accuracy);
strcat(str, tmp);
// Write the result
io_file = fopen(path_out, "wb");
fprintf(io_file, "%.3f\n", accuracy);
for (i = 0; i < IMG_COUNT; i++) {
fprintf(io_file, "%d, %d, %.3f\n", labels_ans[i], labels[i], confidences[i]);
}
fclose(io_file);

// Release the allocated memory and string
(*env)->ReleaseStringUTFChars(env, path_network, nativePath);
}


int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_nsec < y->tv_nsec) {
        int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
        y->tv_nsec -= 1000000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_nsec - y->tv_nsec > 1000000000) {
        int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
        y->tv_nsec += 1000000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
       tv_nsec is certainly positive. */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_nsec = x->tv_nsec - y->tv_nsec;

    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}