// Network.cpp: implementation of the Network class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Network.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iostream>
#include <fcntl.h>
#include <sys/types.h>
#include <string>
using namespace std;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Network::Network()
{
	printf("Network Construction.\n");
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

// This constructor Assumes a properly formatted data file.  It does no checking or corrections.
Network::Network( char* file_name )
{
	int error = 0;

	error = readNetworkFromFile( file_name );  // This function should return an error message for an improperly specified network.
//	setNetworkNeuronActivation( 0.0 );
	if(error == 1)	printf("Bad Network file specification in file %s.\n", file_name);  // if the file is bad print a warning

}


Network::~Network()
{
	//printf("Network destruction.\n");
}

Network net;//Global variable to replace Network::

void instantiateDefaultNetwork( void )
{
	net.numberOfInputs = 2 ;
	net.numberOfOutputs = 2;
	net.numberOfInterNeurons = 6;
	net.networkDimension = net.numberOfInputs + net.numberOfOutputs + net.numberOfInterNeurons;

	net.setNeuronOutput( 0.0 );			// set initial activations to zero. Maybe better to be one.
	net.setNeuronThresholds( 0.0 );		// set thresholds for output in network neurons.
	net.setNeuronLearningRate( 0.0 );	// set learning rate for plastic neurons in network - 0 is non-learning.
	net.setNeuronRefractoryState( 0 );	// set set refactory state for neurons in the network - 0 is no refactory state -- must not be negative.
	net.setNeuronWeightTotal( 1.0 );	// set set inpute weight for neurons in the network - 1 is typical and default-- must not be negative.

	net.setNetworkWeights( 0.5 );		// Temporary function and value to test the code
	net.setNeuronActivation( 0.0 );		// Temporary function and value to test the code
	net.setNetworkOutputs( 0.0 );
	net.setPlasticWeightsMask( 0 );
}

/* --------------------------------------------------

  Print network state

  */
void PrintNetworkState( void )
{
	printf(" Number of inputs: %d\n",net.numberOfInputs);
	printf(" Number of outputs: %d\n",net.numberOfOutputs);
	printf(" Number of interneuorns: %d\n",net.numberOfInterNeurons);
	printf(" Network Dimension: %d\n",net.networkDimension);
}

/* --------------------------------------------------

  networkActivation
  This is a comitted matrix multiplication routine that generates the activation from the current state of the network.

  It treates autapses and inputs specially so it is not strictly speaking pure matrix mathematics
  */

void Network::networkActivation( void  )
{
	int neuron_number, source_neuron_number, k;

	// -----------------  Compute intrisinc network activations

	// -- Update autapses
	for( neuron_number = 0; neuron_number < net.networkDimension; ++neuron_number){

		k = net.networkDimension*neuron_number + neuron_number;  // you should make this a function rather than computing it 2x in this routine, it could be re-used for other routines and avoid problems of different computations in different locations
		net.neuronActivation[neuron_number] = net.neuronActivation[neuron_number] * net.networkWeights[k];
//printf("-- %2.3lf %2.3lf\n", neuronActivation[neuron_number], networkWeights[k]);
	}

	// -- Update inputs from other neurons
	for( neuron_number = 0; neuron_number < net.networkDimension; ++neuron_number){  // note that iterations can be reduced in this routine by skipping the inputs -- which do not need to be updated through network weights, they are set from the outside
		for(source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){

			if(neuron_number != source_neuron_number){						// used self weights above, avoid double dipping
				k = net.networkDimension*source_neuron_number + neuron_number;	// obtain the index of the 2d weight array represented as a 1 d array.
				net.neuronActivation[neuron_number] += net.neuronOutput[source_neuron_number] * net.networkWeights[k];
//printf("-- %2.3lf %2.3lf\n", neuronActivation[neuron_number], networkWeights[k]);
			}
		}

	// ------------------ Add External Inputs to Activations  -----------------------
		if( neuron_number < net.numberOfInputs ) {
			net.neuronActivation[neuron_number] += net.networkInputs[neuron_number]; // Network inputs are set externally
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

	for(i = 0; i< net.numberOfInputs ; i++) {
		net.networkInputs[i] = vector[i];
//printf("input %lf ",vector[i]);
//printf("input %lf ",networkInputs[i]);

	}
//printf("\n ");

}

/* --------------------------------------------------

  copyNetworkInputsToInputNeuronOutputs

*/

void Network::copyNetworkInputsToInputNeuronOutputs( void )
{
	int i;

	for(i = 0; i < net.numberOfInputs; ++i ) {
		net.neuronOutput[i] = net.networkInputs[i];
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
	int i;

	for(i = 0; i< net.numberOfOutputs; ++i) {
		net.networkOutputs[i] = net.neuronOutput[net.numberOfInputs + net.numberOfInterNeurons + i];

//printf("* %d ",numberOfInputs + numberOfInterNeurons + i);
//printf("* %lf ",networkNeuronOutput[numberOfInputs + numberOfInterNeurons -1 + i]);

	}
//printf("\n",networkOutputs[i]);

}

/* --------------------------------------------------

  setNetworkNeuronOutput

*/

//setNetworkNeuronOutput( void )

void Network::copyNeuronActivationsToNeuronOutputs( void )
{
	int i;

	for(i = 0; i < net.networkDimension; ++i){
		net.neuronActivation[i] = net.neuronActivation[i];
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

	for(i = 0; i < net.networkDimension; ++i){
		if(net.neuronActivation[i] > net.neuronThresholds[i]){
			net.neuronOutput[i] = net.neuronActivation[i] - net.neuronThresholds[i];
//			net.neuronActivation[i] = net.neuronActivation[i];
		}
		else net.neuronOutput[i] = 0.0;
//printf("*** %2.3lf %2.3lf\n",neuronOutput[i],neuronThresholds[i]);
	}
//printf("\n ");

}


/* --------------------------------------------------

thresholdNeuronOutputs

Computes a hard thresholded output from the neuron activations using the individual neuron threshold

*/

void Network::squashNeuronOutputs(double max, double slope, double xoffset)
{
	int i;

	for (i = 0; i < net.networkDimension; ++i) {
		if (net.neuronActivation[i] > net.neuronThresholds[i]) {
			net.neuronOutput[i] = net.squashingFunction(net.neuronActivation[i],1.0,1.0,0.0);
			//			net.neuronActivation[i] = net.neuronActivation[i];
		}
		else net.neuronOutput[i] = 0.0;
		//printf("*** %2.3lf %2.3lf\n",neuronOutput[i],neuronThresholds[i]);
	}
	//printf("\n ");
	//squashingFunction(double value, double max, double slope, double xoffset)
}

/* --------------------------------------------------

  getNetworkOuput

  a function meant to supply the network outputs to outside process

*/
void Network::getNetworkOuput( double * vector )
{
	int i;

	for(i = 0; i< net.numberOfOutputs; ++i) {
		vector[i] = net.networkOutputs[i];
	}

}


/* --------------------------------------------------

  setNetworkWeights

  a function meant to supply the network outputs to outside process

*/
void Network::setNetworkWeights( double value )
{
	int i;

	for(i = 0; i< net.networkDimension*net.networkDimension ; ++i) {
		net.networkWeights[i] = value;

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
		net.networkWeights[i*net.networkDimension + i] = value;

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
	int col_start = start_column;

	for(i = start_column; i < end_column; i++) {
		for(j = start_row; j < end_row ; j++) {
			index = net.networkDimension*i +j;
			net.networkWeights[index] = value;

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
				index = net.networkDimension*i +j;
				net.networkWeights[index] = value;
			}
		}
	}

}

//void setNetworkWeightByRow(int row_number, in)
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
				index = net.networkDimension*i +j;
				net.networkWeights[index] = value;
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

	for(i = 0; i < net.networkDimension ; i++) {
		for(j = 0; j < net.networkDimension ; j++) {
			index = net.networkDimension*i +j;
			if( i < j ) net.networkWeights[index] = upper_triangle_value;
			if( i > j ) net.networkWeights[index] = lower_triangle_value;
			if( i == j ) net.networkWeights[index] = diagonal_value;
		}
	}

}

/* --------------------------------------------------
/* --------------------------------------------------

  setPlasticWeightsMask

  a function meant to supply the network outputs to outside process


*/
void Network::setPlasticWeightsMask( short int value )
{
	int i;

	for(i = 0; i< net.networkDimension*net.networkDimension ; ++i) {
		net.plasticWeightsMask[i] = value;

	}


}


/* --------------------------------------------------

  setNetworkNeuronActivation


*/
void Network::setNeuronActivation( double value )
{
	int i;

	for(i = 0; i< net.networkDimension ; ++i) {
		net.neuronActivation[i] = value;

	}

}

/* --------------------------------------------------

  setNetworkNeuronActivation


*/
void Network::setNeuronOutput( double value )
{
	int i;

	for(i = 0; i< net.networkDimension ; ++i) {
		net.neuronOutput[i] = value;

	}


}

/* --------------------------------------------------

  setNetworkNeuronThresholds


*/
void Network::setNeuronThresholds( double value )
{
	int i;

	for(i = 0; i< net.networkDimension ; ++i) {
		net.neuronThresholds[i] = value;

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

	for(i = 0; i< net.networkDimension ; ++i) {
		net.neuronLearningRate[i] = value;

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

	for(i = 0; i< net.networkDimension ; ++i) {
		net.neuronRefractoryState[i] = value;
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

	for(i = 0; i< net.networkDimension ; ++i) {
		net.neuronWeightTotal[i] = value;
	}

}

/* --------------------------------------------------

  setNetworkOutputs


*/
void Network::setNetworkOutputs( double value )
{
	int i;

	for(i = 0; i< net.numberOfOutputs; ++i) {
		net.networkOutputs[i] = value;
	}


}
/* --------------------------------------------------

	squashingFunction

	default value for max, slope, xoffset
	max = 1, slope = 1, xoffset = 0


*/

double Network::squashingFunction(double value, double max, double slope, double xoffset)
{
	double result;
	// result = max / (1 - exp(value * slope + xoffset));
	result = max / (1 + exp(-(value * slope + xoffset)));

	return result;
}


/* --------------------------------------------------

  printNetworkOuput

  a function meant to supply the network outputs to outside process


*/
void Network::printNetworkOuput( void )
{
	int i;

	for(i = 0; i< net.numberOfOutputs; ++i) {
		printf("%f ",net.networkOutputs[i]);
	}
	printf("\n");

}

/* --------------------------------------------------

  cycleNetwork

  a function meant to supply the network outputs to outside process


	Notes:
		1. Inputs  should be set separately. This routine does not make use of external input. It uses the current neuron outputs into inputs.
		The network inputs must be set before calling this routine to get the network to respond to new input
		information.

*/
void Network::cycleNetwork( void )
{

	networkActivation( );						// perform adjusted matrix multiplication of the weights and current network state
//	setNetworkNeuronOutput( );					// Transform activations into outputs and copy
	copyNeuronActivationsToNeuronOutputs( );
	//squashNeuronOutputs(1.0,1.0,0.0);
	thresholdNeuronOutputs( );					// Transform activations into outputs following hard threshold
	setNetworkOuput( );							// Copy the network output to the output array *+* consider moving this call out of the function to allow network "settling time" before external functions have access to the network output

}

void Network::cycleNetworkNormalizeHebbianLearning( void )
{
/* *+*  */
	net.hebbianExcitatoryWeightUpdate( );
	net.normalizeNonDiagonalExcitatoryNeuronWeights( ); // note that this temporary value of 1.0 will ensure the weights of all the units sum to 1.0.
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
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_increment;

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		if(net.neuronLearningRate[target_neuron_number] !=0){ // save clock cyles by only computing updates on weights that are plastic.
			for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
				weight_increment = 0;
				weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
				weight_increment = net.neuronLearningRate[target_neuron_number]*net.neuronOutput[source_neuron_number]*net.neuronOutput[target_neuron_number]*net.plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights
				net.networkWeights[weight_index] += weight_increment;
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
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_increment;

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		if(net.neuronLearningRate[target_neuron_number] !=0){ // save clock cyles by only computing updates on weights that are plastic.
			for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
				weight_increment = 0;
				weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
				if( net.networkWeights[weight_index] > 0 ){

					weight_increment = net.neuronLearningRate[target_neuron_number]*net.neuronOutput[source_neuron_number]*net.neuronOutput[target_neuron_number]*net.plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights
					net.networkWeights[weight_index] += weight_increment;
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

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		if(net.neuronLearningRate[target_neuron_number] !=0){ // save clock cyles by only computing updates on weights that are plastic.
			for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
				weight_increment = 0;
				weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
				if( net.networkWeights[weight_index] < 0 ){
//					weight_increment = neuronLearningRate[target_neuron_number]*neuronOutput[source_neuron_number]*plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights
					weight_increment = net.neuronLearningRate[target_neuron_number]*net.neuronOutput[source_neuron_number]*net.neuronOutput[target_neuron_number]*net.plasticWeightsMask[weight_index];  // remember that the plastic weights mask AND the learning rate for a neuron must agree ( both be non-zero) for a neuron to have adaptive weights

					net.networkWeights[weight_index] -= weight_increment;  // Note that this DECREMENTS the weight
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

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex( source_neuron_number,  target_neuron_number );
//			weight_index = networkDimension*target_neuron_number + source_neuron_number;
			if(net.networkWeights[weight_index] >0) weight_sum += net.networkWeights[weight_index];
			if(net.networkWeights[weight_index] <0) weight_sum = weight_sum - net.networkWeights[weight_index];
		}
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex(  source_neuron_number, target_neuron_number );
//			weight_index = networkDimension*target_neuron_number + source_neuron_number; 			weight_index = networkDimension*source_neuron_number + target_neuron_number;
			net.networkWeights[weight_index] = value*( net.networkWeights[weight_index]/weight_sum);
		}
	}
}



/*
	normalizeNeuronWeights

	Computes the sum of the weights for a given neuron.
	Then make a proportional change in the weights of that neuron so that it sums to the passed value.

	This version preserves the weight total found in a given unit's wieghts.  The function normalizeNeuronWeights( double ) sets them to a user specificed value

	This function use the sum of the absolute value of each weight to determine the normalization value. (Negative weights are inverted in sign before they are summed).

  */
void Network::normalizeNeuronWeights( void )
{
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_sum;

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
				if(net.networkWeights[weight_index] > 0) weight_sum += net.networkWeights[weight_index];
				if(net.networkWeights[weight_index] < 0) weight_sum -= net.networkWeights[weight_index];
		}
			for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
				weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
				net.networkWeights[weight_index] = net.neuronWeightTotal[ target_neuron_number ]*( net.networkWeights[weight_index]/(double)weight_sum);
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

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number){
				if(net.networkWeights[weight_index] > 0) weight_sum += net.networkWeights[weight_index];
				if(net.networkWeights[weight_index] < 0) weight_sum -= net.networkWeights[weight_index];
//printf("Weight sum: %lf ", weight_sum);
			}
		}
		if(target_neuron_number != source_neuron_number && weight_sum != 0.0){  // avoid division by zero for input units the autapse may be the only non-zero weight.
			for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
				weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
				net.networkWeights[weight_index] = net.neuronWeightTotal[ target_neuron_number ]*( net.networkWeights[weight_index]/(double)weight_sum);
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
	int source_neuron_number, target_neuron_number, weight_index;
	double weight_sum;

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && net.networkWeights[weight_index] > 0){
				 weight_sum += net.networkWeights[weight_index];
			}
		}
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && weight_sum != 0.0 && net.networkWeights[weight_index] > 0){  // avoid division by zero for input units the autapse may be the only non-zero weight.
				net.networkWeights[weight_index] = net.neuronWeightTotal[ target_neuron_number ]*( net.networkWeights[weight_index]/weight_sum);
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

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		weight_sum = 0.0;
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && net.networkWeights[weight_index] < 0){
				 weight_sum -= net.networkWeights[weight_index]; //  Since the weights are negative, we want a positve sum for the normalization below. at *+*
			}
		}
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){
			weight_index = net.computeWeightIndex( source_neuron_number, target_neuron_number );
			if(target_neuron_number != source_neuron_number && weight_sum != 0.0 && net.networkWeights[weight_index] < 0){  // avoid division by zero for input units the autapse may be the only non-zero weight.
				net.networkWeights[weight_index] = net.neuronWeightTotal[ target_neuron_number ]*( net.networkWeights[weight_index]/weight_sum); // *+* a postive weight sum here will produce a negative proportion
			}
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
int Network::readNetworkFromFile( char * file_name )
{
	int i, item_count = 0, error = 0;
	char dummy[MAX_DUMMY_STRING_LENGTH];
	FILE *fp;
	fp= fopen(file_name,"r");
	if( fp == 0) error = 1;
	else{
		fscanf(fp,"%s %d",&dummy, &net.numberOfInputs);
		fscanf(fp,"%s %d",&dummy, &net.numberOfOutputs);
		fscanf(fp,"%s %d",&dummy, &net.numberOfInterNeurons);
		fscanf(fp,"%s %d",&dummy, &net.networkDimension); // perhaps networkDimension should be omitted from the Read and computed on the read or in the constructor

	// Read the stored network activations
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension; ++i) fscanf(fp,"%lf",&net.neuronActivation[i]);
	// Read the stored network outputs
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension; ++i) fscanf(fp,"%lf",&net.neuronOutput[i]);
	// Read the stored neuron thresholds
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension; ++i) fscanf(fp,"%lf",&net.neuronThresholds[i]);
	// Read the stored neuron learning rates
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension; ++i) fscanf(fp,"%lf",&net.neuronLearningRate[i]);
	// Read the stored neuron refractory states
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension; ++i) fscanf(fp,"%lf",&net.neuronRefractoryState[i]);
	// Read the stored neuron refractory states
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension; ++i) fscanf(fp,"%lf",&net.neuronWeightTotal[i]);
	// Read the stored network weights
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension*net.networkDimension; ++i) fscanf(fp,"%lf",&net.networkWeights[i]);
	// Read the stored network inputs
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.numberOfInputs; ++i) fscanf(fp,"%lf",&net.networkInputs[i]);
		fscanf(fp,"%s",&dummy);
	// Read the stored network outputs
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.numberOfOutputs; ++i) fscanf(fp,"%lf",&net.networkOutputs[i]);
	// Read the stored network plastic weights mask
		fscanf(fp,"%s",&dummy);
		for( i = 0 ; i < net.networkDimension*net.networkDimension; ++i) fscanf(fp,"%d",&net.plasticWeightsMask[i]);
	}

	fclose(fp);
	return(error);
}

/* --------------------------------------------------

`File

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
/*  This function overwrites a matrix weight in your file.
	You must specify which weight you want to write by specifing the row and column
*/
int Network::writeNetworkToFile( char * file_name)
{



	int i, item_count = 0, error = 0;
	FILE *fp;
	fp= fopen(file_name,"w");

	if( fp == 0) error = 1;
	else{

		fprintf(fp,"net.numberOfInputs %d\n",net.numberOfInputs);
		fprintf(fp,"net.numberOfOutputs %d\n",net.numberOfOutputs);
		fprintf(fp,"numberOfInterNeurons %d\n",net.numberOfInterNeurons);
		fprintf(fp,"networkDimension %d\n",net.networkDimension); // perhaps this should be omitted from the write and computed on the read or in the constructor

	// Write the stored network activations
		fprintf(fp,"networkActivations\n");
		for( i = 0 ; i < net.networkDimension; ++i) fprintf(fp,"%lf ",net.neuronActivation[i]);
		fprintf(fp,"\n");
	// Write the stored network outputs
		fprintf(fp,"networkOutputs\n");
		for( i = 0 ; i < net.networkDimension; ++i) fprintf(fp,"%lf ",net.neuronOutput[i]);
		fprintf(fp,"\n");
	// Write the stored network thresholds
		fprintf(fp,"neuronThreshold\n");
		for( i = 0 ; i < net.networkDimension; ++i) fprintf(fp,"%lf ",net.neuronThresholds[i]);
		fprintf(fp,"\n");
	// Write the stored neuron learning rates
		fprintf(fp,"neuronLearningRate\n");
		for( i = 0 ; i < net.networkDimension; ++i) fprintf(fp,"%lf ",net.neuronLearningRate[i]);
		fprintf(fp,"\n");
	// Write the stored neuron refractory states
		fprintf(fp,"neuronRefactoryState\n");
		for( i = 0 ; i < net.networkDimension; ++i) fprintf(fp,"%lf ",net.neuronRefractoryState[i]);
		fprintf(fp,"\n");
	// Write the stored neuron weight total
		fprintf(fp,"neuronWeightTotal\n");
		for( i = 0 ; i < net.networkDimension; ++i) fprintf(fp,"%lf ",net.neuronWeightTotal[i]);
		fprintf(fp,"\n");
	// Write the stored network weights

		fprintf(fp,"networkweights\n");
		item_count = 0;		// Set up counter for number of rows printed
		for( i = 0 ; i < net.networkDimension*net.networkDimension; ++i){
			fprintf(fp,"%lf ",net.networkWeights[i]);
			++item_count;
			if(item_count == net.networkDimension){
				fprintf(fp,"\n");       // place a new line after each row printed to make reading of the output file intuitive
				item_count = 0;
			}
		}
		fprintf(fp,"\n");

	// Write the stored network inputs
		fprintf(fp,"networkinputs\n");
		for( i = 0 ; i < net.numberOfInputs; ++i) fprintf(fp,"%lf ",net.networkInputs[i]);
		fprintf(fp,"\n");
	// Write the stored network outputs
		fprintf(fp,"networkoutputs\n");
		for( i = 0 ; i < net.numberOfOutputs; ++i) fprintf(fp,"%lf ",net.networkOutputs[i]);
		fprintf(fp,"\n");
	// Write the stored plastic weights mask
		fprintf(fp,"networkplasticweightsmask\n");
		item_count = 0;		// Set up counter for number of rows printed
		for( i = 0 ; i < net.networkDimension*net.networkDimension; ++i){
			fprintf(fp,"%d ",net.plasticWeightsMask[i]);
			++item_count;
			if(item_count == net.networkDimension){  // place a new line after each row printed to make reading of the output file intuitive
				fprintf(fp,"\n");
				item_count = 0;
			}
		}
		fprintf(fp,"\n");

	}


	fclose(fp);
	return(error);
}
/*  pick and choose witch weights you want, every other weight is set to zero  */
int Network::writeNetworkToFile(char * file_name, double alpha, double beta, double gamma, double delta, double epsilon, double eta)
{

	cout << "!! STARTING A NEW TRIAL!!" << endl;

	int i, item_count = 0, error = 0;
	FILE *fp;
	fp = fopen(file_name, "w");

	if (fp == 0) error = 1;
	else {

		fprintf(fp, "numberOfInputs %d\n", net.numberOfInputs);
		fprintf(fp, "numberOfOutputs %d\n", net.numberOfOutputs);
		fprintf(fp, "numberOfInterNeurons %d\n", net.numberOfInterNeurons);
		fprintf(fp, "networkDimension %d\n", net.networkDimension); // perhaps this should be omitted from the write and computed on the read or in the constructor

		// Write the stored network activations
		fprintf(fp, "networkActivations\n");
		for (i = 0; i < net.networkDimension; ++i) fprintf(fp, "%lf ", net.neuronActivation[i]);
		fprintf(fp, "\n");
		// Write the stored network outputs
		fprintf(fp, "networkOutputs\n");
		for (i = 0; i < net.networkDimension; ++i) fprintf(fp, "%lf ", net.neuronOutput[i]);
		fprintf(fp, "\n");
		// Write the stored network thresholds
		fprintf(fp, "neuronThreshold\n");
		for (i = 0; i < net.networkDimension; ++i) fprintf(fp, "%lf ", net.neuronThresholds[i]);
		fprintf(fp, "\n");
		// Write the stored neuron learning rates
		fprintf(fp, "neuronLearningRate\n");
		for (i = 0; i < net.networkDimension; ++i) fprintf(fp, "%lf ", net.neuronLearningRate[i]);
		fprintf(fp, "\n");
		// Write the stored neuron refractory states
		fprintf(fp, "neuronRefactoryState\n");
		for (i = 0; i < net.networkDimension; ++i) fprintf(fp, "%f ", 0.0);
		fprintf(fp, "\n");
		// Write the stored neuron weight total
		fprintf(fp, "neuronWeightTotal\n");
		for (i = 0; i < net.networkDimension; ++i) fprintf(fp, "%lf ", net.neuronWeightTotal[i]);
		fprintf(fp, "\n");
		// Write the stored network weights

		fprintf(fp, "networkweights\n");
		int column_count = 0;
		int row_count = 0;// Set up counter for number of rows printed
		for (i = 0; i < net.networkDimension*net.networkDimension; ++i) {
			if ((row_count == 3 && column_count == 0) || (row_count == 4 && column_count == 0)) {
				fprintf(fp, "%lf ", alpha);
			}
			else if ((row_count == 1 && column_count == 1) || (row_count == 2 && column_count == 2)) {
				fprintf(fp, "%lf ", beta);
			}
			else if ((row_count == 3 && column_count == 3) || (row_count == 4 && column_count == 4)) {
				fprintf(fp, "%lf ", gamma);
			}
			else if ((row_count == 3 && column_count == 1) || (row_count == 4 && column_count == 2)) {
				fprintf(fp, "%lf ", delta);
			}
			else if ((row_count == 1 && column_count == 3) || (row_count == 2 && column_count == 4)) {
				fprintf(fp, "%f ", epsilon);
			}
			else if ((row_count == 4 && column_count == 3) || (row_count == 3 && column_count == 4)) {
				fprintf(fp, "%f ", eta);
			}
			else {
				fprintf(fp, "%lf ", 0);
			}
			++column_count;

			if (column_count == net.networkDimension) {
				fprintf(fp, "\n");       // place a new line after each row printed to make reading of the output file intuitive
				column_count = 0;
				++row_count; // Network dimension has bee reached, startnew row.
			}
		}
		fprintf(fp, "\n");


		// Write the stored network inputs
		fprintf(fp, "networkinputs\n");
		for (i = 0; i < net.numberOfInputs; ++i) fprintf(fp, "%lf ", net.networkInputs[i]);
		fprintf(fp, "\n");
		// Write the stored network outputs
		fprintf(fp, "networkoutputs\n");
		for (i = 0; i < net.numberOfOutputs; ++i) fprintf(fp, "%f ", 0.0);
		fprintf(fp, "\n");
		// Write the stored plastic weights mask
		fprintf(fp, "networkplasticweightsmask\n");
		item_count = 0;		// Set up counter for number of rows printed
		for (i = 0; i < net.networkDimension*net.networkDimension; ++i) {
			fprintf(fp, "%d ", net.plasticWeightsMask[i]);
			++item_count;

			if (item_count == net.networkDimension) {  // place a new line after each row printed to make reading of the output file intuitive
				fprintf(fp, "\n");
				item_count = 0;
			}
		}
		fclose(fp);
		return(error);
	}
}
/* --------------------------------------------------

writeNetworkActivationStateToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
void Network::writeNetworkActivationStateToFile( char * file_name )
{
	int i;
	FILE *fp;
	fp= fopen(file_name,"a");

	for( i=0 ; i < net.networkDimension; ++i){

		fprintf(fp,"%lf ",  net.neuronActivation[i]);
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
void Network::writeNetworkOutputStateToFile(char * file_name)
{
	int i;
	FILE *fp;

	fp= fopen(file_name,"a");

	for( i=0 ; i < net.networkDimension; ++i){

		//std::cout << "Current cycle: " << currentCycle << std::endl;
		fprintf(fp,"%lf ",  net.neuronOutput[i]);

	}

	fprintf(fp,"\n");
	fclose(fp);
}


/* Added By Roberto Coyotl This function writes only the ouputs of your network to a file
 * Added for the purpose of speeding up the Paramaterized trial study for the wilson oscillator.
 */
void Network::writeNetworkOuputToFile(char *fileName)
{
	FILE *fp;
	fp = fopen(fileName , "a");
	for (int i = 0; i< net.numberOfOutputs; ++i) {
		fprintf( fp, "%f ", net.networkOutputs[i]);
	}
	fprintf(fp,"\n");
	fclose(fp);
}

void Network::writeTrialInfo(char* fileName, int currentTrial) {// Added by Roberto Coyotl
	FILE *fp;

	fp = fopen(fileName, "a");
	fprintf(fp, "Your current trial number is: ");
	fprintf(fp, "%d", currentTrial);
	fprintf(fp, "\n");
	fclose(fp);
}
/* --------------------------------------------------

writeNetworkSquashedOutputToFile

takes as input a file name
writes the file to be formatted according to the standard network form
changes to this should be mirrored in readNetworkFromFile

*/
void Network::writeNetworkSquashedOutputStateToFile(char * file_name)
{
	int i;
	FILE *fp;

	fp = fopen(file_name, "a");

	for (i = 0; i < net.networkDimension; ++i) {

		fprintf(fp, "%lf ", net.squashingFunction(net.neuronOutput[i], 1,1, 0));

	}

	fprintf(fp, "\n");
	fclose(fp);
}


/* --------------------------------------------------

writeNetworkActivationStateToFile

	takes as input a file name
   writes the file to be formatted according to the standard network form
   changes to this should be mirrored in readNetworkFromFile

*/
void Network::writeNetworkWeightsToFile( char * file_name )
{
	int source_neuron_number, target_neuron_number, weight_index;
	FILE *fp;
	fp= fopen(file_name,"a");

	fprintf(fp," | "); // pretty printing to delinate the start of a print of the weight set

	for( target_neuron_number = 0; target_neuron_number < net.networkDimension; ++target_neuron_number){
		for( source_neuron_number = 0; source_neuron_number < net.networkDimension; ++source_neuron_number){

		weight_index = net.computeWeightIndex(source_neuron_number, target_neuron_number);

		fprintf(fp,"%lf ",  net.networkWeights[weight_index]);

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

	for( i=0 ; i < net.networkDimension; ++i){

		printf("%3.3lf ",  net.neuronOutput[i]);
	}

	printf("\n");

}

void Network::printNetworkSquashedOutputState(void)
{
	int i;

	for (i = 0; i < net.networkDimension; ++i) {

		printf("%3.3lf ", net.squashingFunction(net.neuronOutput[i], 1, 1 ,0));
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
	return( net.networkDimension*source_neuron_number + target_neuron_number );
}



