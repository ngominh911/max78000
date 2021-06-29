/*
 * ann.h
 *
 *  Created on: 24 Jun 2021
 *      Author: DucminhN
 */

#ifndef ANN_H_
#define ANN_H_

#define MAX_SIZE  32
#define LAYER_SIZE 3
//Declarations ANN structure
typedef struct networks{
    int n_layers;
    int dim[LAYER_SIZE];
    int weights[LAYER_SIZE][MAX_SIZE][MAX_SIZE];
    int biases[LAYER_SIZE][MAX_SIZE];
}network;


//Function prototypes

double sigmoid(double x);
double RELU(double x);
double ACTIVATE(double x);
void init_ann_with_weights(network*,int[],int [LAYER_SIZE][MAX_SIZE][MAX_SIZE], int[LAYER_SIZE][MAX_SIZE],int);
void feed_forward(network*,double output[LAYER_SIZE][MAX_SIZE]);
//void train(network*, double**,int,double);
int predict(network*,double[MAX_SIZE]);
void test(network*,double**,int);

//Misc Functions
void arrayCopy(double dest[],double source[],int length);

//Definitions



void init_ann_with_weights(network* ann,int dim[],int weights[LAYER_SIZE][MAX_SIZE][MAX_SIZE],int biases[LAYER_SIZE][MAX_SIZE],int n_layers){

    ann->n_layers = n_layers;
    for(int i=0;i<n_layers;i++){
        ann->dim[i] = dim[i];
    }
    for(int i=1;i<n_layers;i++){
        for(int j=0;j<dim[i];j++){
            for(int k=0;k<dim[i-1];k++){
                ann->weights[i-1][j][k] = weights[i-1][j][k];
            }
        }
    }
    for(int i=0;i<n_layers;i++){
        for(int j=0;j<ann->dim[i];j++){
            if(i == 0){
                ann->biases[i][j] = 0;
            }
            else{
                ann->biases[i][j] = biases[i][j];
            }
        }
    }
}





int predict(network* ann, double data[MAX_SIZE]){
    double output[ann->n_layers][MAX_SIZE];
    arrayCopy(output[0],data,ann->dim[0]);
    feed_forward(ann,output);
    int maxval = 0;
    if(ann->dim[ann->n_layers-1] == 1){
        if(output[ann->n_layers-1][0] >= 0.5){
            return 1;
        }
        else{
            return 0;
        }
    }
    else{
        for(int i=0;i<ann->dim[ann->n_layers-1];i++){

            if(output[ann->n_layers-1][i] > output[ann->n_layers - 1][maxval]){
                maxval = i;
            }
        }
        return maxval;
    }
}




void feed_forward(network* ann,double output[LAYER_SIZE][MAX_SIZE]){
//    double* input;
    for(int i=1;i<ann->n_layers;i++){
//        input = output[i-1];
        for(int j=0;j<ann->dim[i];j++){
            double f_sum = 0;
            for(int k=0;k<ann->dim[i-1];k++){
            	//if(i==1){
            		//f_sum += ann->weights[i-1][j][k] * output[i-1][k];
            	//}
            	 f_sum += ann->weights[i-1][j][k] * output[i-1][ann->dim[i-1] - k -1];
            }
            f_sum += ann->biases[i][j];
            if(i == 1){
            	output[i][j] = RELU(f_sum);
            }
            else output[i][j] = f_sum;
//            printf("output[%d][%d] = %f \n",i,j,output[i][j]);

        }
    }
}

double RELU(double x){
	if(x>0) return x; // RELU
	return 0;
}

double sigmoid(double x){
    return 1/ (1 + exp(-x));
}

double ACTIVATE(double value){

	//int limit = (1 << 32)-1;

	int result = 0;
	int factorA = 0x0444eb90;
	int temp_result = factorA*value;

	temp_result = temp_result >> 18;


	temp_result = temp_result >> 22;

	if (temp_result < 0)
		result = 0;
	else
		result = temp_result;

	return result;
}



//Misc Functions


void arrayCopy(double dest[],double source[],int length){
    for(int i=0;i<length;i++){
        dest[i] = source[i];
    }
}



#endif /* ANN_H_ */
