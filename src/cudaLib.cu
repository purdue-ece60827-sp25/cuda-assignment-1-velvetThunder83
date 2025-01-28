
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
// ===================================================================================================================
// Below this line is the code for part 1 of this assignemnt. 

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		y[index] += scale * x[index];
	}
}

int runGpuSaxpy(int vectorSize) {
	std::cout << "Hello GPU Saxpy!\n";
	uint64_t vectorBytes = vectorSize * sizeof(float);
	float scale = 2.0f;

	// =======================================================================
	// The intial part of the code is for creating 2 arrays on the CPU>
	float *a, *b, *c;
	
	a = (float *) malloc(vectorBytes);
	b = (float *) malloc(vectorBytes);

	// this is used to copy the results from the GPU to the CPU.
	c = (float *) malloc(vectorBytes);

	// initalize only the first 2 vectors. 
	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);

	// =======================================================================
	float *d, *e;

	cudaMalloc(&d, vectorBytes);
	cudaMalloc(&e, vectorBytes);
	cudaMemcpy(d, a, vectorBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(e, b, vectorBytes, cudaMemcpyHostToDevice);

	// =======================================================================
	// call the actual GPU kernel.
	saxpy_gpu<<<ceil(vectorSize/256.0),256>>>(d, e, scale, vectorSize);

	// =======================================================================
	// copy the code back from the GPU to host CPU. The results are compared as such. 
	cudaMemcpy(c, e, vectorBytes, cudaMemcpyDeviceToHost);
	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// ======================================================================= 
	// need to free all the allocated space. 
	cudaFree(d);
	cudaFree(e);
	free(a);
	free(b);
	free(c);

	return 0;
}
// ===================================================================================================================
// Below this line is the code for part 2 of this assignemnt. 

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	// generate the thread index. 
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// =======================================================================
	// if the thread index is greater than the expected size, 
	// then you don't exeucte that thread. 
	if (index >= pSumSize) {
		return;
	}

	// =======================================================================
	// set up a random number generator. 
	curandState_t rng;
	curand_init(clock64(), index, 0, &rng);

	// =======================================================================
	// generate a whole bunch of x and y numbers
	// if x ^ 2 + y ^ 2 <= 1.0f, then you increment the hits.
	// curand_uniform returns numebrs between 0 and 1. 
	uint64_t hits = 0; 
	for(int i = 0; i < sampleSize; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);

		if(x * x + y * y <= 1.0f) {
			hits++;
		}
	}

	// =======================================================================
	// each thread has 1 spot in the pSums Array to commit the results to. 
	pSums[index] = hits;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	// this gives you the index you will write the sum to in the totals array.  
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// have to handle odd cases where the reduce size and pSums are not neat multiples. 
	if (index >= (pSumSize + reduceSize - 1) / reduceSize) {
		return;
	}

	uint64_t start = reduceSize * index;
	uint64_t sum = 0; 
	for (uint64_t i = 0; i < reduceSize; ++i) {
		if(start + i < pSumSize) {
			sum += pSums[start + i];
		}
	}
	totals[index] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << std::setprecision(10);
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	double approxPi = 0;
	uint64_t pSumSize = generateThreadCount; 
	uint64_t vectorBytes = generateThreadCount * sizeof(uint64_t);
	uint64_t totalSumBytes = reduceThreadCount * sizeof(uint64_t);

	uint64_t * totalSums = (uint64_t *) malloc(totalSumBytes); 
	// uint64_t * pSums = (uint64_t *) malloc(vectorBytes); 
	uint64_t * totalSums_gpu; 
	uint64_t * pSums_gpu; 

	cudaMalloc(&pSums_gpu, vectorBytes);
	cudaMalloc(&totalSums_gpu, totalSumBytes);

	cudaMemset(&pSums_gpu, 0, vectorBytes);
	cudaMemset(&totalSums_gpu, 0, totalSumBytes);

	generatePoints<<<ceil(generateThreadCount/256.0), 256>>>(pSums_gpu, pSumSize, sampleSize);
	reduceCounts<<<ceil(reduceThreadCount/256.0), 256>>>(pSums_gpu, totalSums_gpu, pSumSize, reduceSize);
	// cudaDeviceSynchronize();

	cudaMemcpy(totalSums, totalSums_gpu, totalSumBytes, cudaMemcpyDeviceToHost);
	// cudaMemcpy(pSums, pSums_gpu, vectorBytes, cudaMemcpyDeviceToHost);

	// uint64_t totalHitCount = 0;
	// for(int i = 0; i < pSumSize; i++) {
	// 	totalHitCount += pSums[i];
	// }


	uint64_t totalHitCount = 0;
	for(int i = 0; i < reduceThreadCount; i++) {
		totalHitCount += totalSums[i];
	}

	approxPi = ((float)totalHitCount / sampleSize) / generateThreadCount;
	approxPi = approxPi * 4.0f;

	// Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
