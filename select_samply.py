import cupy as cp
import numpy as np
import pandas as pd
import argparse
import scipy.linalg
from utilpy3 import load_cifar
np.set_printoptions(threshold=10000)

samples = 10
train_sample_type = 5
train_samples = 2

parser = argparse.ArgumentParser(description = 'Convolutional Neural Tangent Kernel (CNTK) for CIFAR-10')
parser.add_argument('--depth', default = 21, type = int, help = 'depth of CNTK (#conv layers + 1)')
parser.add_argument('--gap', default = "yes", type = str, help = 'whether GAP (global average pooling) is used')
parser.add_argument('--fix', default = "yes", type = str, help = 'whether first layer and last layer are fixed (or trained) (see Section 4.2 in our paper)')
args = parser.parse_args()

d = args.depth
gap = (args.gap == "yes")
fix = (args.fix == "yes")

#CUDA kernel for convolution operation
conv3 = cp.RawKernel(r'''
extern "C" __global__
void conv3(const float s[32][32][32][32], float t[32][32][32][32])
{
	int x1 = threadIdx.x + blockIdx.x - 31;
	int y1 = threadIdx.y + blockIdx.y - 31;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;

	__shared__ float d[32 + 2][32 + 2];
	if (x2 == 0){
		d[0][y2 + 1] = d[33][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][33] = d[33][0] = d[33][33] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][33] = 0;
	}

	if (x1 < 0 || x1 > 31 || y1 < 0 || y1 > 31){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	__syncthreads();

	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];

}''', 'conv3')

conv3check = cp.RawKernel(r'''
extern "C" __global__
void conv3check(const float s[32][32][32][32], float t[32][32][32][32], float D[32 + 2][32 + 2]) 
{
	int x1 = threadIdx.x + blockIdx.x - 31;
	int y1 = threadIdx.y + blockIdx.y - 31;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;

	__shared__ float d[32 + 2][32 + 2];
	if (x2 == 0){
		d[0][y2 + 1] = d[33][y2 + 1] = 0;
		if (x2 == 0 && y2 == 0)
			d[0][0] = d[0][33] = d[33][0] = d[33][33] = 0; 
	}
	if (y2 == 0){
		d[x2 + 1][0] = d[x2 + 1][33] = 0;
	}

	if (x1 < 0 || x1 > 31 || y1 < 0 || y1 > 31){
		d[x2 + 1][y2 + 1] = 0;
		return;
	}
	else
		d[x2 + 1][y2 + 1] = s[x1][y1][x2][y2];
	  D[x2 + 2][y2 + 2] = s[x1][y1][x2][y2];
	__syncthreads();


	t[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
					  + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
					  + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];

}''', 'conv3check')

conv_blocks = (63, 63)
conv_threads = (32, 32)

#CUDA kernel for activation
trans = cp.RawKernel(r'''
extern "C" __global__
void trans(float s[32][32][32][32], float t[32][32][32][32], const float l[32][32], const float r[32][32], const float il[32][32], const float ir[32][32])
{
	int x1 = blockIdx.x;
	int y1 = blockIdx.y;
	int x2 = threadIdx.x + ((blockIdx.z >> 2) << 3);
	int y2 = threadIdx.y + ((blockIdx.z & 3) << 3);
	float S = s[x1][y1][x2][y2], T = t[x1][y1][x2][y2], L = l[x1][y1], R = r[x2][y2], iL = il[x1][y1], iR = ir[x2][y2];
	S = S * iL * iR;
	float BS = (S * (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) + sqrtf(1.0f - min(S * S, 1.0f))) * L * R / 28.274333882308138f;
	S = (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) / 28.274333882308138;
	t[x1][y1][x2][y2] = T*S*S + T * S + BS;
	s[x1][y1][x2][y2] = BS;

}''', 'trans')
trans_blocks = (32, 32, 16)
trans_threads = (8, 8)

#Calculate diagonal entries of $\Sigma^{(h)}(x, x)$ and their reciprocals. See Section 4.3 in our paper. 
def xx(x):
	RL = [1.0, ]
	iRL = [1.0, ]

	S = cp.matmul(x.T, x).reshape(32, 32, 32, 32)
	D = cp.zeros((34, 34), dtype = cp.float32)
	#print("before",S[:][0][0][0])
	#print("before",D)
	conv3check(conv_blocks, conv_threads, (S, S, D))
	#print("after",S[:][0][0][0])
	#print("after",D)
	T = cp.zeros((32, 32, 32, 32), dtype = cp.float32)
	if not fix:
		T += S

	for i in range(1, d - 1):
		#cupy.diag only take diagonal array output length 1024
		#cupy.sqrt Elementwise square root
		L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
		iL = 1.0 / L
		RL.append(L)
		iRL.append(iL)
		trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))

	L = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
	iL = 1.0 / L
	RL.append(L)
	iRL.append(iL)
	trans(trans_blocks, trans_threads, (S, T, L, L, iL, iL))	
	
	if fix:
		T -= S
	return RL, iRL

#Caclulate the kernel value of x and z.
#Lx and Lz are diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
#iLx and iLz are reciprocals of diagonal entries of $\Sigma^{(h)}(x, x)$ and $\Sigma^{(h)}(z, z)$. 
def xz(x, z, Lx, Lz, iLx, iLz):
	S = cp.matmul(x.T, z).reshape(32, 32, 32, 32)
	conv3(conv_blocks, conv_threads, (S, S))
	T = cp.zeros((32, 32, 32, 32), dtype = cp.float32)
	if not fix:
		T += S

	for i in range(1, d - 1):
		trans(trans_blocks, trans_threads, (S, T, Lx[i], Lz[i], iLx[i], iLz[i]))		
		conv3(conv_blocks, conv_threads, (S, S))
		conv3(conv_blocks, conv_threads, (T, T))

	trans(trans_blocks, trans_threads, (S, T, Lx[-1], Lz[-1], iLx[-1], iLz[-1]))
	if fix:
		T -= S
	#cp.mean(cp.linalg.eigh(T.reshape(1024, 1024))[0])
	return cp.mean(T) if gap else cp.trace(T.reshape(1024, 1024))



def calculate(it):
	#Load CIFAR-10.
	(X_train, y_train), (X_test, y_test) = load_cifar()
	deadlist = []
	x = 0
	for index,item in enumerate(y_train):
		if item==it:
			x = x + 1
			deadlist.append(index)
		if x >= samples:
			break

	for it in range(train_sample_type):
		it = it + 2
		x = 0
		for index,item in enumerate(y_train):
			if item==it:
				x = x + 1
				deadlist.append(index)
			if x >= train_samples:
				break

	X_train = X_train[deadlist,:,:,:]
	y_train = y_train[deadlist]



	X = X_train
	N = X.shape[0]
	N_train = X_train.shape[0]

	X = cp.asarray(X).reshape(-1, 3, 1024)
	print(X.shape)

	#Calculate diagonal entries.
	L = []
	iL = []
	for i in range(N):
		Lx, iLx = xx(X[i])	
		L.append(Lx)
		iL.append(iLx)

	#####Calculate kernel values.
	#####Below we provide a naive implementation using for-loops.
	#####Parallelize this part according to your specific computing enviroment to utilize multiple GPUs.
	H = {}
	compare = []
	compares = []
	for i in range(N):
		sum=0
		items = []
		for j in range(N):
			keyname = ""
			sorts = [i,j]
			sorts.sort()
			for item in sorts:
				keyname = keyname + str(item) + ":"
			if keyname[:-1] not in H.keys():
				H[keyname[:-1]] = xz(X[i], X[j], L[i], L[j], iL[i], iL[j])
			print(H)  

calculate(0)      
