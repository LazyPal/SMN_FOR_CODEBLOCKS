// Network.cpp: implementation of the Network class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Network.h"
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Network::Network()
{
	printf("Network Construction.\n");
	openLogFile();
	numberOfInputs = 2 ;
	numberOfOutputs = 2;
	numberOfInterNeurons = 6;
	networkDimension = numberOfInputs + numberOfOutputs + numberOfInterNeurons;

	setNeuronOutput( 0.0 );			// set initial activations to zero. Maybe better to be one.
	setNeuronThresholds( 0.0 );		// set thresholds for output in network neurons.
	setNeuronLearningRate( 0.0 );	// set learning rate for plastic neurons in network - 0 is non-learning.
	setNeuronRefractoryState( 0 );	// set set refactory state for neurons in the network - 0 is no refactory state -- must not be negative.
	setNeuronWeightTotal( 1.0 );	// set set refactory state for neurons in the network - 0 is no refactory state -- must not be negative.
	setNetworkWeights( 0.5 );		// Temporary function and value to test the code
	setNeuronActivation( 0.0 );		// Temporary function and value to test the code
	setNetworkOutputs( 0.0 );
	setPlasticWeightsMask( 0 );
	normalizeNeuronWeights( );		// Set the sum of network weights for each unit to the unit total specified above.
}


// This is a blank constructor that will produce a file in the current standard format.
Network::Network(int inputs, int interneurons, int outputs, char * out_file_name )
{
	printf("Network Construction.\n");
	openLogFile();
	numberOfInputs = inputs ;
	numberOfOutputs = outputs;
	numberOfInterNeurons = interneurons;
	networkDimension = numberOfInputs + numberOfOutputs + numberOfInterNeurons;

	setNeuronOutput( 0.0 );			// set initial activations to zero. Maybe better to be one.
	setNeuronThresholds( 0.0 );		// set thresholds for output in network neurons.
	setNeuronLearningRate( 0.0 );	// set learning rates for network neurons -- 0 means that neuron does not adjust its connection weights
	setNeuronRefractoryState( 0 );	// set set refactory state for neurons in the network - 0 is no refactory state -- must not be negative.
	setNeuronWeightTotal( 1.0 );		// set set weight for neurons in the network - 1.0 typical and default.
	setNetworkWeights( 0.5 );		// Temporary function and value to test the code
	setNeuronActivation( 0.0 );		// Temporary function and value to test the code
	setNetworkOutputs( 0.0 );
	setPlasticWeightsMask( 0 );
	normalizeNeuronWeights( );		// Set the sum of network weights for each unit to the unit total specified above.

	writeNetworkToFile( out_file_name );
}

Network::Network(std::string file_name) {
	openLogFile();
	int error = 0;

	error = readNetworkFromFile( file_name );  // This function should return an error message for an improperly specified network.
//	setNetworkNeuronActivation( 0.0 );
	if(error == 1)	printf("Bad Network file specification in file %s.\n", file_name.c_str());  // if the file is bad print a warning
}
/*
// This constructor Assumes a properly formatted data file.  It does no checking or corrections.
Network::Network( char * file_name )
{
	openLogFile();
	int error = 0;

	error = readNetworkFromFile( file_name );  // This function should return an error message for an improperly specified network.
//	setNetworkNeuronActivation( 0.0 );
	if(error == 1)	printf("Bad Network file specification in file %s.\n", file_name);  // if the file is bad print a warning
}
*/

Network::~Network()
{
	printf("Network destruction.\n");
}

bool Network::openLogFile()
{
	printf("opening log file...\n");
    logFile = fopen("log.txt", "w");
	return logFile != NULL;
}

void Network::closeLogFile()
{
	fclose(logFile);
}

FILE* Network::getLogFile() {
	return logFile;
}

void Network::instantiateDefaultNetwork( void )
{
	numberOfInputs = 2 ;
	numberOfOutputs = 2;
	numberOfInterNeurons = 6;
	networkDimension = numberOfInputs + numberOfOutputs + numberOfInterNeurons;

	setNeuronOutput( 0.0 );			// set initial activations to zero. Maybe better to be one.
	setNeuronThresholds( 0.0 );		// set thresholds for output in network neurons.
	setNeuronLearningRate( 0.0 );	// set learning rate for plastic neurons in network - 0 is non-learning.
	setNeuronRefractoryState( 0 );	// set set refactory state for neurons in the network - 0 is no refactory state -- must not be negative.
	setNeuronWeightTotal( 1.0 );	// set set inpute weight for neurons in the network - 1 is typical and default-- must not be negative.

	setNetworkWeights( 0.5 );		// Temporary function and value to test the code
	setNeuronActivation( 0.0 );		// Temporary function and value to test the code
	setNetworkOutputs( 0.0 );
	setPlasticWeightsMask( 0 );
}

/* --------------------------------------------------

  Print network state

  */
void Network::PrintNetworkState( void )
{
	printf(" Number of inputs: %d\n",numberOfInputs);
	printf(" Number of outputs: %d\n",numberOfOutputs);
	printf(" Number of interneurons: %d\n",numberOfInterNeurons);
	printf(" Network Dimension: %d\n",networkDimension);
	printNetworkWeights();
}

void Network::printNetworkWeights() {

	printf(" Network weights:\n");
	int item_count = 0;
	printf("  ");
	int i;
	for (i = 0 ; i < networkDimension; ++i)	{
		printf("    %2d", i + 1);
	}
	printf("\n  ");
	for (i = 0 ; i < networkDimension; ++i)	{
		printf("------");
	}
	printf("\n");
	for (i = 0 ; i < networkDimension*networkDimension; ++i) {
		if (item_count == 0) {
			printf("%2d|", (i / networkDimension) + 1);
		}
		printf("% .2f ", networkWeights[i]);
		++item_count;
		if (item_count == networkDimension) {
			printf("\n");
			item_count = 0;
		}
	}
}

/*
 * Return a copy of the given matrix.
 */
double* copyMatrix(double* matrix, const int dimension) {
	static double* copy = new double[dimension * dimension];
	std::copy(matrix, matrix + (dimension * dimension), copy);
	return copy;
}

void printDifferences(double* beforeMatrix, double* afterMatrix, int dimension) {
	int item_count = 0;
	printf("  ");
	int i;
	for (i = 0 ; i < dimension; ++i)	{
		printf("    %2d", i + 1);
	}
	printf("\n  ");
	for (i = 0 ; i < dimension; ++i)	{
		printf("------");
	}
	printf("\n");
	for (i = 0 ; i < dimension*dimension; ++i) {
		if (item_count == 0) {
			printf("%2d|", (i / dimension) + 1);
		}
		printf("% .2f ", afterMatrix[i] - beforeMatrix[i]);
		++item_count;
		if (item_count == dimension) {
			printf("\n");
			item_count = 0;
		}
	}
}


/* --------------------------------------------------

  networkActivation
  This is a comitted matrix multiplication routine that generates the activation from the current state of the network.

  It treates autapses and inputs specially so it is not strictly speaking pure matrix mathematics
  */

void Network::networkActivation( void  )
{
	fprintf(logFile, "1. networkActivation()\n");
	int neuron_number, source_neuron_number, k;

	// -----------------  Compute intrinsic network activations

	fprintf(logFile, "  a. Update autapses\n");
	// -- Update autapses
	for( neuron_number = 0; neuron_number < networkDimension; ++neuron_number){

		k = networkDimension*neuron_number + neuron_number;  // you should make this a function rather than computing it 2x in this routine, it could be re-used for other routines and avoid problems of different computations in different locations
        fprintf(logFile, "activation[%d] = self(%.4f) * weight[%d,%d](%.4f)", neuron_number+1,
               neuronActivation[neuron_number], neuron_number+1, neuron_number+1, networkWeights[k]);
		neuronActivation[neuron_number] = neuronActivation[neuron_number] * networkWeights[k];
        fprintf(logFile, " => %.4f\n", neuronActivation[neuron_number]);
	}

    fprintf(logFile, "  b. Update inputs from other neurons\n");
	// -- Update inputs from other neurons
	for( neuron_number = 0; neuron_number < networkDimension; ++neuron_number){  // note that iterations can be reduced in this routine by skipping the inputs -- which do not need to be updated through network weights, they are set from the outside
		for(source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){

			if(neuron_number != source_neuron_number){						// used self weights above, avoid double dipping
				k = networkDimension*source_neuron_number + neuron_number;	// obtain the index of the 2d weight array represented as a 1 d array.
				if (networkWeights[k] != 0) {
                	fprintf(logFile, "activation[%d] = self(%.4f) + output[%d](%.4f) * %.4f",
                    	   neuron_number+1, neuronActivation[neuron_number], source_neuron_number+1, neuronOutput[source_neuron_number], networkWeights[k]);
				}
				neuronActivation[neuron_number] += neuronOutput[source_neuron_number] * networkWeights[k];
				if (networkWeights[k] != 0) {
	                fprintf(logFile, " => %.4f\n", neuronActivation[neuron_number]);
				}
			}
		}

	// ------------------ Add External Inputs to Activations  -----------------------
		if (neuron_number < numberOfInputs ) {
            fprintf(logFile, "activation[%d] = self(%.4f) + network_input[%d]=%.4f", neuron_number+1,
                   neuronActivation[neuron_number], neuron_number+1, networkInputs[neuron_number]);
			neuronActivation[neuron_number] += networkInputs[neuron_number]; // Network inputs are set externally
            fprintf(logFile, " => %.4f\n", neuronActivation[neuron_number]);
//printf("-- %2.3lf %2.3lf\n", neuronActivation[neuron_number], networkInputs[neuron_number]);
		}
	}
}

/* --------------------------------------------------

  setNetworkInput

  copy and external vector of inputs the the input section of the network inputs vector.

	  note: this routine should check that the inputs are approprirate -- in bounds.

*/

void Network::setNetworkInput( double *vector)
{
	int i;

	for(i = 0; i< numberOfInputs ; ++i) {
		networkInputs[i] = vector[i];
//printf("input %d <= %lf\n",i, vector[i]);
//printf("input %lf ",networkInputs[i]);

	}
//printf("\n ");

}

void Network::setNetworkInput(vector<double> values)
{
	int i;

	for(i = 0; i< numberOfInputs ; ++i) {
		networkInputs[i] = values[i];
	}
}

/* --------------------------------------------------

  copyNetworkInputsToInputNeuronOutputs

*/

void Network::copyNetworkInputsToInputNeuronOutputs( void )
{
	int i;

	for(i = 0; i < numberOfInputs; ++i ) {
		neuronOutput[i] = networkInputs[i];
//printf("input %lf ",networkNeuronOutput[i]);

	}
//printf("\n ");

}

/* --------------------------------------------------

 setNetworkOuput

  copy and external vector of inputs the the input section of the network inputs vector.

	  note: this routine should check that the inputs are approprirate -- in bounds.

*/

void Network::setNetworkOuput( void )
{
    fprintf(logFile, "4. setNetworkOutput()\n");
	int i;

	for(i = 0; i< numberOfOutputs; ++i) {
		networkOutputs[i] = neuronOutput[numberOfInputs + numberOfInterNeurons + i];
        fprintf(logFile, "networkOutput[%d] := neuronOutput[%d](%.4f)\n", i+1,
               numberOfInputs + numberOfInterNeurons + i+1,
               neuronOutput[numberOfInputs + numberOfInterNeurons + i]);
//printf("* %d ",numberOfInputs + numberOfInterNeurons + i);
//printf("* %lf ",networkNeuronOutput[numberOfInputs + numberOfInterNeurons -1 + i]);

	}
//printf("\n",networkOutputs[i]);

}

/* --------------------------------------------------

  setNetworkNeuronOutput

*/

//Network::setNetworkNeuronOutput( void )

void Network::copyNeuronActivationsToNeuronOutputs( void )
{
    fprintf(logFile, "2. copyNeuronActivationsToNeuronOutputs() (WARN) does nothing\n");
	int i;

    // FIXME SL: This doesn't do anything; should it copy them to neuronOutput array?
	for(i = 0; i < networkDimension; ++i){
		neuronActivation[i] = neuronActivation[i];
//printf("%2.2lf %2.2lf | ",networkNeuronOutput[i] , networkNeuronActivation[i]);
	}
//printf("\n");

}

/* --------------------------------------------------

  thresholdNeuronOutputs

  Computes a hard thresholded output from the neuron activations using the individual neuron threshold

*/

void Network::thresholdNeuronOutputs( void )
{
	int i;

    fprintf(logFile, "3. thresholdNeuronOutputs()\n");
	for(i = 0; i < networkDimension; ++i){
		if(neuronActivation[i] > neuronThresholds[i]){
			neuronOutput[i] = neuronActivation[i] - neuronThresholds[i];
//			neuronActivation[i] = neuronActivation[i];
		}
		else neuronOutput[i] = 0.0;
        fprintf(logFile, "neuronOutput[%d]: activation(%.4f) > threshold(%.4f)? ==> %.4f\n", i+1,
               neuronActivation[i], neuronThresholds[i], neuronOutput[i]);
	}
//printf("\n ");

}


/* --------------------------------------------------

  Network::squashNeuronOutputs( double offset, double expSlope)

  Computes a transformed logistic function (squashed) output from the neuron activations using the individual neuron threshold
		if no offset: offset == 0
		if no slope change: expSlope == 1

*/

void Network::squashNeuronOutputs( double offset=0, double expSlope=1)
{
	int i;

	for(i = 0; i < networkDimension; ++i){
			neuronOutput[i] = (1/(1+exp(-neuronActivation[i] * expSlope) ) + offset);

	}
}

/* --------------------------------------------------

  getNetworkOuput

  a function meant to supply the network outputs to outside process

*/
void Network::getNetworkOuput( double * vector )
{
	int i;

	for(i = 0; i< numberOfOutputs; ++i) {
		vector[i] = networkOutputs[i];
	}

}

double* Network::getNetworkWeights() {
	return networkWeights;
}

int Network::getNetworkDimension() {
	return networkDimension;
}

int Network::getNumInputs() {
	return numberOfInputs;
}

int Network::getNumOutputs() {
	return numberOfOutputs;
}


/* --------------------------------------------------

  setNetworkWeights

  a function meant to supply the network outputs to outside process

*/
void Network::setNetworkWeights( double value )
{
	int i;

	for(i = 0; i< networkDimension*networkDimension ; ++i) {
		networkWeights[i] = value;

	}
}


/* --------------------------------------------------

  setNetworkWeightsDiagonalRange

  a function meant to supply the network outputs to outside process


*/
void Network::setNetworkWeightsDiagonalRange( double value, int start_row_and_col, int end_row_and_col )
{
	int i;

	for(i = start_row_and_col; i < end_row_and_col; i++) {
		networkWeights[i*networkDimension + i] = value;

	}

}

/* --------------------------------------------------

  setNetworkWeightsRectangle

  a function meant to supply the network outputs to outside process

  NEEDS to check that values passeed are in bounds of the array.
*/
void Network::setNetworkWeightsRectangle( double value, int start_row, int end_row, int start_column, int end_column )
{
	int i, j, index;

	for(i = start_column; i < end_column; i++) {
		for(j = start_row; j < end_row ; j++) {
			index = networkDimension*i +j;
			networkWeights[index] = value;
		}
	}

}

/* --------------------------------------------------

  setNetworkWeightsUpperTriangle

  a function meant to supply the network outputs to outside process

  NEEDS to check that values passeed are in bounds of the array.
*/
void Network::setNetworkWeightsUpperTriangle( double value, int start_row, int end_row, int start_column, int end_column )
{
	int i, j, index;

	for(i = start_column; i < end_column; i++) {
		for(j = start_row; j < end_row ; j++) {
			if(i > j ){
				index = networkDimension*i +j;
				networkWeights[index] = value;
			}
		}
	}

}

/* --------------------------------------------------

  setNetworkWeightsLowerTriangle

  a function meant to supply the network outputs to outside process

  NEEDS to check that values passeed are in bounds of the array.
*/
void Network::setNetworkWeightsLowerTriangle( double value, int start_row, int end_row, int start_column, int end_column )
{
	int i, j, index;

	for(i = start_column; i < end_column; i++) {
		for(j = start_row; j < end_row ; j++) {
			if(i > j ){
				index = networkDimension*i +j;
				networkWeights[index] = value;
			}
		}
	}

}


/* --------------------------------------------------

  setNetworkWeightsUpperLowerTriangleAndDiagonal

  a function meant to supply the network outputs to outside process
  NEEDS to check that values passeed are in bounds of the array.

*/
void Network::setNetworkWeightsUpperLowerTriangleAndDiagonal( double diagonal_value, double upper_triangle_value, double lower_triangle_value)
{
	int i, j, index;

	for(i = 0; i < networkDimension ; i++) {
		for(j = 0; j < networkDimension ; j++) {
			index = networkDimension*i +j;
			if( i < j ) networkWeights[index] = upper_triangle_value;
			if( i > j ) networkWeights[index] = lower_triangle_value;
			if( i == j ) networkWeights[index] = diagonal_value;
		}
	}

}

/* --------------------------------------------------

  setPlasticWeightsMask

  a function meant to supply the network outputs to outside process


*/
void Network::setPlasticWeightsMask( short int value )
{
	int i;

	for(i = 0; i< networkDimension*networkDimension ; ++i) {
		plasticWeightsMask[i] = value;

	}


}


/* --------------------------------------------------

  setNetworkNeuronActivation


*/
void Network::setNeuronActivation( double value )
{
	int i;

	for(i = 0; i< networkDimension ; ++i) {
		neuronActivation[i] = value;

	}

}

/* --------------------------------------------------

  setNetworkNeuronActivation


*/
void Network::setNeuronOutput( double value )
{
	int i;

	for(i = 0; i< networkDimension ; ++i) {
		neuronOutput[i] = value;

	}


}

/* --------------------------------------------------

  setNetworkNeuronThresholds


*/
void Network::setNeuronThresholds( double value )
{
	int i;

	for(i = 0; i< networkDimension ; ++i) {
		neuronThresholds[i] = value;

	}


}

/* --------------------------------------------------

  setNetworkLearningRate

  Set to zero for non-learning neurons
  Negative values should be used with care.
  Typically positive values within the interval [0,1]


*/
void Network::setNeuronLearningRate( double value )
{
	int i;

	for(i = 0; i< networkDimension ; ++i) {
		neuronLearningRate[i] = value;

	}


}


/* --------------------------------------------------

  setNetworkRefractoryState

  Integer values are the number of the time-steps that the neuron is unable to fire.
  Set to zero for non-refactory neurons
  Must not be negative.   Negative values are set to zero.  !!! SET TO ZERO !!!
  Returns an error of 1 if the refractory state has been changed to zero because a negative value was passed as an argument

*/
int Network::setNeuronRefractoryState( int value )
{
	int i, error = 0;

	if(value <0){
		value = 0;
		error = 1;
	}

	for(i = 0; i< networkDimension ; ++i) {
		neuronRefractoryState[i] = value;
	}

	return(error);

}



/* --------------------------------------------------

  setNeuronWeightTotal

  Integer values are the number of the time-steps that the neuron is unable to fire.
  Set to zero for non-refactory neurons
  Must not be negative.   Negative values are set to zero.  !!! SET TO ZERO !!!
  Returns an error of 1 if the refractory state has been changed to zero because a negative value was passed as an argument

*/
void Network::setNeuronWeightTotal( double value)
{
	int i;

	for(i = 0; i< networkDimension ; ++i) {
		neuronWeightTotal[i] = value;
	}

}

/* --------------------------------------------------

  setNetworkOutputs


*/
void Network::setNetworkOutputs( double value )
{
	int i;

	for(i = 0; i< numberOfOutputs; ++i) {
		networkOutputs[i] = value;
	}


}


/* --------------------------------------------------

  printNetworkOuput

  a function meant to supply the network outputs to outside process


*/
void Network::printNetworkOuput( void )
{
	int i;

	for(i = 0; i< numberOfOutputs; ++i) {
		printf("%f ",networkOutputs[i]);
	}
	printf("\n");

}

/* --------------------------------------------------

  cycleNetwork

  a function meant to supply the network outputs to outside process
  thresholds the outputs (no output below or equal to threshold)


	Notes:
		1. Inputs  should be set separately. This routine does not make use of external input. It uses the current neuron outputs into inputs.
		The network inputs must be set before calling this routine to get the network to respond to new input
		information.

*/
void Network::cycleNetwork( void )
{
    fprintf(logFile, "***** cycleNetwork\n");
	networkActivation( );						// perform adjusted matrix multiplication of the weights and current network state
//	setNetworkNeuronOutput( );					// Transform activations into outputs and copy
	copyNeuronActivationsToNeuronOutputs( );
	thresholdNeuronOutputs( );					// Transform activations into outputs following hard threshold
	setNetworkOuput( );							// Copy the network output to the output array *+* consider moving this call out of the function to allow network "settling time" before external functions have access to the network output

}




/* --------------------------------------------------

  cycleNetworkSquash( double offset=0, double expSlope=1  )

  a function meant to supply the network outputs to outside process
  squashes the outputs


	Notes:
		1. Inputs  should be set separately. This routine does not make use of external input. It uses the current neuron outputs into inputs.
		The network inputs must be set before calling this routine to get the network to respond to new input
		information.

*/
void Network::cycleNetworkSquash(  double offset=0, double expSlope=1 )
{

	networkActivation( );						// perform adjusted matrix multiplication of the weights and current network state
//	setNetworkNeuronOutput( );					// Transform activations into outputs and copy
	copyNeuronActivationsToNeuronOutputs( );
	squashNeuronOutputs(offset,expSlope );					// Transform activations into outputs following hard threshold
	setNetworkOuput( );							// Copy the network output to the output array *+* consider moving this call out of the function to allow network "settling time" before external functions have access to the network output

}








void Network::cycleNetworkNormalizeHebbianLearning( void )
{
    fprintf(logFile, "***** cycleNetworkNormalizeHebbianLearning()\n");
/* *+*  */
	hebbianExcitatoryWeightUpdate( );
	normalizeNonDiagonalExcitatoryNeuronWeights( ); // note that this temporary value of 1.0 will ensure the weights of all the units sum to 1.0.
}


/*
	hebbianWeightUpdate

  NOTE: weight updates will only occur if the weights are marked as plastic and if the neuron has a non-zero learning rate
  Both these parameters must agree.  If the learning rate is zero there will be no update for that neuron.
  If the plastic weights mask is zero there will be no weight update for that weight.

  NOTE: The weight changes effected by this routine will cause the values of the weights to grow.  Over many cycles they will grow without  bound.
  Some form of normalization or negative weights need to be used to offset this unbounded weight growth if the network is to be stable

  */
void Network::hebbianWeightUpdate( void  )
{
    fprintf(logFile, "1. hebbianWeightUpdate()\n");
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_increment;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		if(neuronLearningRate[target_neuron_number] !=0){ // save clock cyles by only computing updates on weights that are plastic.
			for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
				weight_increment = 0;
				weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
				weight_increment = neuronLearningRate[target_neuron_number]*neuronOutput[source_neuron_number]*neuronOutput[target_neuron_number]*plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights
                fprintf(logFile, "weight[%d] = prev_weight(%.4f) + %.4f", weight_index+1,
                       networkWeights[weight_index], weight_increment);
				networkWeights[weight_index] += weight_increment;
                fprintf(logFile, " => %.4f\n", networkWeights[weight_index]);
			}

		}
	}
}

/*
	hebbianPositiveWeightUpdate

  Same as hebbianWeightUpdate but applied only to positive valued weights

  NOTE: weight updates will only occur if the weights are marked as plastic and if the neuron has a non-zero learning rate
  Both these parameters must agree.  If the learning rate is zero there will be no update for that neuron.
  If the plastic weights mask is zero there will be no weight update for that weight.

  NOTE: The weight changes effected by this routine will cause the values of the weights to grow.  Over many cycles they will grow without  bound.
  Some form of normalization or negative weights need to be used to offset this unbounded weight growth if the network is to be stable

  */
void Network::hebbianExcitatoryWeightUpdate( void )
{
    fprintf(logFile, "1. hebbianExcitatoryWeightUpdate()\n");
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_increment;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		if(neuronLearningRate[target_neuron_number] !=0){ // save clock cyles by only computing updates on weights that are plastic.
			for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
				weight_increment = 0;
				weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
				if( networkWeights[weight_index] > 0 ){

					weight_increment = neuronLearningRate[target_neuron_number]*neuronOutput[source_neuron_number]*neuronOutput[target_neuron_number]*plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights
                    fprintf(logFile, "weight[%d,%d](%.4f) += learning_rate(%.4f) * output[%d](%.4f) * output[%d](%.4f) * mask(%d)",
							source_neuron_number+1, target_neuron_number+1,
							networkWeights[weight_index],
                           neuronLearningRate[target_neuron_number],
							source_neuron_number+1,
                           neuronOutput[source_neuron_number],
							target_neuron_number+1,
                           neuronOutput[target_neuron_number],
                           plasticWeightsMask[weight_index],
							weight_increment);
					networkWeights[weight_index] += weight_increment;
                    fprintf(logFile, "  => %.4f\n", networkWeights[weight_index]);
				}
			}

		}
	}
}


/*
	hebbianInhibitoryWeightUpdate

  Same as hebbianWeightUpdate but applied only to negative valued weights

  This function DECREMENTS the weights, making them more negative

  NOTE: weight updates will only occur if the weights are marked as plastic and if the neuron has a non-zero learning rate
  Both these parameters must agree.  If the learning rate is zero there will be no update for that neuron.
  If the plastic weights mask is zero there will be no weight update for that weight.

  NOTE: The weight changes effected by this routine will cause the values of the weights to grow.  Over many cycles they will grow without  bound.
  Some form of normalization or negative weights need to be used to offset this unbounded weight growth if the network is to be stable

  */
void Network::hebbianInhibitoryWeightUpdate( void )
{
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_increment;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		if(neuronLearningRate[target_neuron_number] !=0){ // save clock cyles by only computing updates on weights that are plastic.
			for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
				weight_increment = 0;
				weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
				if( networkWeights[weight_index] < 0 ){
//					weight_increment = neuronLearningRate[target_neuron_number]*neuronOutput[source_neuron_number]*plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights
					weight_increment = neuronLearningRate[target_neuron_number]*neuronOutput[source_neuron_number]*neuronOutput[target_neuron_number]*plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights

					networkWeights[weight_index] -= weight_increment;  // Note that this DECREMENTS the weight
				}
			}

		}
	}
}
/*
	normalizeNeuronWeights

	Computes the sume of the weights for a given neuron.
	Then make a proportional change in the weights of that neuron so that it sums to the passed value.

  */
void Network::normalizeNeuronWeights( double value )
{
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_sum;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex( source_neuron_number,  target_neuron_number );
//			weight_index = networkDimension*target_neuron_number + source_neuron_number;
			if(networkWeights[weight_index] >0) weight_sum += networkWeights[weight_index];
			if(networkWeights[weight_index] <0) weight_sum = weight_sum - networkWeights[weight_index];
		}
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex(  source_neuron_number, target_neuron_number );
//			weight_index = networkDimension*target_neuron_number + source_neuron_number; 			weight_index = networkDimension*source_neuron_number + target_neuron_number;
			networkWeights[weight_index] = value*( networkWeights[weight_index]/weight_sum);
		}
	}
}



/*
	normalizeNeuronWeights

	Computes the sume of the weights for a given neuron.
	Then make a proportional change in the weights of that neuron so that it sums to the passed value.

	This version preserves the weight total found in a given unit's wieghts.  The function normalizeNeuronWeights( double ) sets them to a user specificed value

	This function use the sum of the absolute value of each weight to determine the normalization value. (Negative weights are inverted in sign before they are summed).

  */
void Network::normalizeNeuronWeights( void )
{
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_sum;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
				if(networkWeights[weight_index] > 0) weight_sum += networkWeights[weight_index];
				if(networkWeights[weight_index] < 0) weight_sum -= networkWeights[weight_index];
		}
			for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
				weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
				networkWeights[weight_index] = neuronWeightTotal[ target_neuron_number ]*( networkWeights[weight_index]/(double)weight_sum);
		}
	}
}


/*
	normalizeNonDiagonalNeuronWeights

	Computes the sume of the weights for a given neuron.
	Then make a proportional change in the weights of that neuron so that it sums to the passed value.
	It does not update weights that are autapses.  self connections along the network diagonal.

	This function leaves the weight matrix diagonals ( autapses ) unchanged. And it does not use them in the calculations.


  */
void Network::normalizeNonDiagonalNeuronWeights( void )
{
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_sum;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number){
				if(networkWeights[weight_index] > 0) weight_sum += networkWeights[weight_index];
				if(networkWeights[weight_index] < 0) weight_sum -= networkWeights[weight_index];
//printf("Weight sum: %lf ", weight_sum);
			}
		}
		if(target_neuron_number != source_neuron_number && weight_sum != 0.0){  // avoid division by zero for input units the autapse may be the only non-zero weight.
			for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
				weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
				networkWeights[weight_index] = neuronWeightTotal[ target_neuron_number ]*( networkWeights[weight_index]/(double)weight_sum);
			}
		}
	}
}

/*
	normalizeNonDiagonalExcitatoryNeuronWeights

	Computes the sume of the weights for a given neuron.
	Then make a proportional change in the weights of that neuron so that it sums to the passed value.
	It does not update weights that are autapses.  self connections along the network diagonal.

	This function leaves the weight matrix diagonals ( autapses ) unchanged. And it does not use them in the calculations.


  */
void Network::normalizeNonDiagonalExcitatoryNeuronWeights( void )
{
    fprintf(logFile, "2. normalizeNonDiagonalExcitatoryNeuronWeights()\n");
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_sum;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && networkWeights[weight_index] > 0){
				 weight_sum += networkWeights[weight_index];
			}
		}
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && weight_sum != 0.0 && networkWeights[weight_index] > 0){  // avoid division by zero for input units the autapse may be the only non-zero weight.
                fprintf(logFile, "weight[%d,%d] = weight_total[%d](%.4f) * weight[%d,%d](%.4f) / sum(%.4f)",
                       source_neuron_number+1, target_neuron_number+1, target_neuron_number+1,
                       neuronWeightTotal[ target_neuron_number ],
                       source_neuron_number+1, target_neuron_number+1,
                       networkWeights[weight_index], weight_sum);
				networkWeights[weight_index] = neuronWeightTotal[ target_neuron_number ]*( networkWeights[weight_index]/weight_sum);
                fprintf(logFile, " => %.4f\n", networkWeights[weight_index]);
			}
		}
	}
}

/*
	normalizeNonDiagonalExcitatoryNeuronWeights

	Computes the sume of the weights for a given neuron.
	Then make a proportional change in the weights of that neuron so that it sums to the passed value.
	It does not update weights that are autapses.  self connections along the network diagonal.

	This function leaves the weight matrix diagonals ( autapses ) unchanged. And it does not use them in the calculations.



  */
void Network::normalizeNonDiagonalInhibitoryNeuronWeights( void )
{

	int source_neuron_number, target_neuron_number, weight_index;
	double weight_sum;

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && networkWeights[weight_index] < 0){
				 weight_sum -= networkWeights[weight_index]; //  Since the weights are negative, we want a positve sum for the normalization below. at *+*
			}
		}
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){
			weight_index = computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && weight_sum != 0.0 && networkWeights[weight_index] < 0){  // avoid division by zero for input units the autapse may be the only non-zero weight.
				networkWeights[weight_index] = neuronWeightTotal[ target_neuron_number ]*( networkWeights[weight_index]/weight_sum); // *+* a postive weight sum here will produce a negative proportion
			}
		}
	}
}

/*
 * Update the weight between two neurons to the given value. Neurons are 1-indexed.
 */
void Network::updateWeight(int from_neuron, int to_neuron, double new_weight) {
	// Neuron indicies are 1-based, so subtract 1 to be 0-based
	int weight_index = computeWeightIndex(from_neuron - 1, to_neuron - 1);

	fprintf(logFile, "Update weight: %2d => %2d: %.4f => %.4f\n", from_neuron+1, to_neuron+1,
		networkWeights[weight_index], new_weight);
	networkWeights[weight_index] = new_weight;
}

void Network::resetNeuronOutputs() {
	int i;
	for (i=0; i < networkDimension; ++i) {
		neuronOutput[i] = 0.0;
	}
}

/*
 Read the values from the file at the given file pointer and set each one into
 the appropriate array index. If a value is not found in the file, set it to the given
 default value
 */
void Network::readRowFromFile(FILE* fp, short* array, short defaultVal) {
	for( int i = 0 ; i < networkDimension; ++i) {
		int items = fscanf(fp,"%lf",&array[i]);
		if (items == 0) {
			array[i] = defaultVal;
		}
	}
}

/*
 Read the values from the file at the given file pointer and set each one into
 the appropriate array index. If a value is not found in the file, set it to the given
 default value
 */
void Network::readMultipleRowsFromFile(FILE* fp, short* array, short defaultVal) {
	for( int i = 0 ; i < networkDimension*networkDimension; ++i) {
		int items = fscanf(fp,"%lf",&array[i]);
		if (items == 0) {
			array[i] = defaultVal;
		}
	}

}

/*
 Read the values from the file at the given file pointer and set each one into
 the appropriate array index. If a value is not found in the file, set it to the given
 default value
 */
void Network::readRowFromFile(FILE* fp, double* array, double defaultVal) {
	for( int i = 0 ; i < networkDimension; ++i) {
		int items = fscanf(fp,"%lf",&array[i]);
		if (items == 0) {
			array[i] = defaultVal;
		}
	}
}

/*
 Read the values from the file at the given file pointer and set each one into
 the appropriate array index. If a value is not found in the file, set it to the given
 default value
 */
void Network::readMultipleRowsFromFile(FILE* fp, double* array, double defaultVal) {
	for( int i = 0 ; i < networkDimension*networkDimension; ++i) {
		int items = fscanf(fp,"%lf",&array[i]);
		if (items == 0) {
			array[i] = defaultVal;
		}
	}

}

/* --------------------------------------------------

readNetworkFromFile

  takes as input a file name
  expects the file to be formatted according to the standard network form
   changes to this should be mirrored in writeNetworkToFile
   returns an error message 1 if there was an error, 0 if there was no error on file open.

*/
int Network::readNetworkFromFile( std::string file_name )
{
    printf("Reading file %s\n", file_name.c_str());
	int i, error = 0;
	char dummy[MAX_DUMMY_STRING_LENGTH];
	FILE *fp;
	fp = fopen(file_name.c_str(),"r");
	if (fp == NULL) error = 1;
	else{
		fscanf(fp,"%s %d",&dummy, &numberOfInputs);
		fscanf(fp,"%s %d",&dummy, &numberOfOutputs);
		fscanf(fp,"%s %d",&dummy, &numberOfInterNeurons);
		fscanf(fp,"%s %d",&dummy, &networkDimension); // perhaps networkDimension should be omitted from the Read and computed on the read or in the constructor

	// Read the stored network activations
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, neuronActivation, 0.0);
	// Read the stored network outputs
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, neuronOutput, 0.0);
	// Read the stored neuron thresholds
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, neuronThresholds, 0.0);
	// Read the stored neuron learning rates
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, neuronLearningRate, 0.0);
	// Read the stored neuron refractory states
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, neuronRefractoryState, 0);
	// Read the stored neuron refractory states
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, neuronWeightTotal, 1.0);
	// Read the stored network weights
		fscanf(fp,"%s",&dummy);
		readMultipleRowsFromFile(fp, networkWeights, 0.0);
//		for( i = 0 ; i < networkDimension*networkDimension; ++i) fscanf(fp,"%lf",&networkWeights[i]);
	// Read the stored network inputs
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, networkInputs, 0.0);
		fscanf(fp,"%s",&dummy);
	// Read the stored network outputs
		fscanf(fp,"%s",&dummy);
		readRowFromFile(fp, networkOutputs, 0.0);
	// Read the stored network plastic weights mask
		fscanf(fp,"%s",&dummy);
		readMultipleRowsFromFile(fp, plasticWeightsMask, 0.0);
//		for( i = 0 ; i < networkDimension*networkDimension; ++i) fscanf(fp,"%d",&plasticWeightsMask[i]);
	}

	fclose(fp);
	return(error);
}

/* --------------------------------------------------

writeNetworkToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
int Network::writeNetworkToFile( std::string file_name )
{
	int i, item_count = 0, error = 0;
	FILE *fp;
	fp= fopen(file_name.c_str(),"w");

	if( fp == 0) error = 1;
	else{
		fprintf(fp,"numberOfInputs %d\n",numberOfInputs);
		fprintf(fp,"numberOfOutputs %d\n",numberOfOutputs);
		fprintf(fp,"numberOfInterNeurons %d\n",numberOfInterNeurons);
		fprintf(fp,"networkDimension %d\n",networkDimension); // perhaps this should be omitted from the write and computed on the read or in the constructor

	// Write the stored network activations
		fprintf(fp,"networkActivations\n");
		for( i = 0 ; i < networkDimension; ++i) fprintf(fp,"%lf ",neuronActivation[i]);
		fprintf(fp,"\n");
	// Write the stored network outputs
		fprintf(fp,"networkOutputs\n");
		for( i = 0 ; i < networkDimension; ++i) fprintf(fp,"%lf ",neuronOutput[i]);
		fprintf(fp,"\n");
	// Write the stored network thresholds
		fprintf(fp,"neuronThreshold\n");
		for( i = 0 ; i < networkDimension; ++i) fprintf(fp,"%lf ",neuronThresholds[i]);
		fprintf(fp,"\n");
	// Write the stored neuron learning rates
		fprintf(fp,"neuronLearningRate\n");
		for( i = 0 ; i < networkDimension; ++i) fprintf(fp,"%lf ",neuronLearningRate[i]);
		fprintf(fp,"\n");
	// Write the stored neuron refractory states
		fprintf(fp,"neuronRefactoryState\n");
		for( i = 0 ; i < networkDimension; ++i) fprintf(fp,"%lf ",neuronRefractoryState[i]);
		fprintf(fp,"\n");
	// Write the stored neuron weight total
		fprintf(fp,"neuronWeightTotal\n");
		for( i = 0 ; i < networkDimension; ++i) fprintf(fp,"%lf ",neuronWeightTotal[i]);
		fprintf(fp,"\n");
	// Write the stored network weights
		fprintf(fp,"networkweights\n");
		item_count = 0;		// Set up counter for number of rows printed
		for( i = 0 ; i < networkDimension*networkDimension; ++i){
			fprintf(fp,"%lf ",networkWeights[i]);
			++item_count;
			if(item_count == networkDimension){
				fprintf(fp,"\n");       // place a new line after each row printed to make reading of the output file intuitive
				item_count = 0;
			}
		}
		fprintf(fp,"\n");
	// Write the stored network inputs
		fprintf(fp,"networkinputs\n");
		for( i = 0 ; i < numberOfInputs; ++i) fprintf(fp,"%lf ",networkInputs[i]);
		fprintf(fp,"\n");
	// Write the stored network outputs
		fprintf(fp,"networkoutputs\n");
		for( i = 0 ; i < numberOfOutputs; ++i) fprintf(fp,"%lf ",networkOutputs[i]);
		fprintf(fp,"\n");
	// Write the stored plastic weights mask
		fprintf(fp,"networkplasticweightsmask\n");
		item_count = 0;		// Set up counter for number of rows printed
		for( i = 0 ; i < networkDimension*networkDimension; ++i){
			fprintf(fp,"%d ",plasticWeightsMask[i]);
			++item_count;
			if(item_count == networkDimension){  // place a new line after each row printed to make reading of the output file intuitive
				fprintf(fp,"\n");
				item_count = 0;
			}
		}
		fprintf(fp,"\n");
	}

	fclose(fp);
	return(error);
}


/* --------------------------------------------------

writeNetworkActivationStateToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
void Network::writeNetworkActivationStateToFile( std::string file_name )
{
	int i;
	FILE *fp;
	fp= fopen(file_name.c_str(),"a");

	for( i=0 ; i < networkDimension; ++i){

		fprintf(fp,"%lf ",  neuronActivation[i]);
	}
	fprintf(fp,"\n");

	fclose(fp);
}


void Network::writeNetworkInputToFile( std::string file_name ) {
	int i;
	FILE *fp;

	fp= fopen(file_name.c_str(),"a");

	for( i=0 ; i < numberOfInputs; ++i){

		fprintf(fp,"%lf ",  networkInputs[i]);

	}

	fprintf(fp,"\n");
	fclose(fp);
}

/* --------------------------------------------------

writeNetworkOutputToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
void Network::writeNetworkOutputStateToFile( std::string file_name )
{
	int i;
	FILE *fp;

	fp= fopen(file_name.c_str(),"a");

	for( i=0 ; i < networkDimension; ++i){

		fprintf(fp,"%lf ",  neuronOutput[i]);

	}

	fprintf(fp,"\n");
	fclose(fp);
}

/* --------------------------------------------------

writeNetworkActivationStateToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
void Network::writeNetworkWeightsToFile( std::string file_name )
{
	int source_neuron_number, target_neuron_number, weight_index;
	FILE *fp;
	fp= fopen(file_name.c_str(),"a");

	fprintf(fp," | "); // pretty printing to delinate the start of a print of the weight set

	for( target_neuron_number = 0; target_neuron_number < networkDimension; ++target_neuron_number){
		for( source_neuron_number = 0; source_neuron_number < networkDimension; ++source_neuron_number){

		weight_index = computeWeightIndex(source_neuron_number, target_neuron_number);

		fprintf(fp,"%lf ",  networkWeights[weight_index]);

		}
		fprintf(fp," * "); // pretty printing to delinate the weights of the different units.
	}
	fprintf(fp," |\n"); // line break to mark the end of each call to this function ( tyically the time-step as this function was intended )

	fclose(fp);
}

/* --------------------------------------------------

writeNetworkOutputToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
void Network::printNetworkOutputState( void )
{
	int i;

	for( i=0 ; i < networkDimension; ++i){

		printf("%3.3lf ",  neuronOutput[i]);
	}

	printf("\n");

}

/* --------------------------------------------------

writeNetworkOutputToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
int Network::computeWeightIndex( int source_neuron_number, int target_neuron_number )
{
	return( networkDimension*source_neuron_number + target_neuron_number );
}
