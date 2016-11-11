import warnings
warnings.filterwarnings("ignore")
import random
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import math

INF = 999999
worst_mean = 0
gamma = 1975	

def L2(a,b):
	return pow((a[0]-b[0]),2) + pow((a[1]-b[1]),2)

def RBF(a,b):
	return math.exp(-gamma*(L2(a,b)))

def kmeans(blob_X):
	K = [2,3,5]
	for k in K:
		mean = []
		for i in range(0,k):
			x1 = random.uniform(-1.5,1.5)
			x2 = random.uniform(-1.5,1.5)
			mean.append(np.array([x1,x2]))
		represented_by_old = []
		represented_by = [worst_mean for i in range(0,blob_X.shape[0])]
		count = 0
		while represented_by != represented_by_old and count < 100:
			count += 1
			point_count_for_mean = [0 for i in range(0,k)]
			represented_by_old = represented_by
			represented_by = [worst_mean for i in range(0,blob_X.shape[0])]
			for i in range(0,blob_X.shape[0]):
				best_dist = INF
				best_mean = worst_mean
				for j in range(0,k):
					z = np.linalg.norm(blob_X[i]-mean[j])
					dist = z
					if dist < best_dist:
						best_dist = dist
						best_mean = j
						represented_by[i] = j
			
			for i in range(0,blob_X.shape[0]):
				point_count_for_mean[represented_by[i]] += 1
			new_mean = [np.array([0,0]) for i in range(0,k)]
			for i in range(0,blob_X.shape[0]):
				new_mean[represented_by[i]] = new_mean[represented_by[i]] + blob_X[i]
			for i in range(0,k):
				if point_count_for_mean[i] != 0.0:
					new_mean[i] = new_mean[i]/point_count_for_mean[i]
				else:
					new_mean[i] = mean[i]
			mean = new_mean
		plt.scatter(blob_X[:,0], blob_X[:,1],c = represented_by)
		plt.show()


def kernelkmeans(blob_X):
	k=2
	mean = [0 for i in range(0,2)]
	for i in range(0,k):
		
		p1 = random.randint(0,blob_X.shape[0]-1)
		p2 = random.randint(0,blob_X.shape[0]-1)
		mean[0] = blob_X[p1]
		mean[1] = blob_X[p2]
	represented_by_old = []
	represented_by = [worst_mean for i in range(0,blob_X.shape[0])]
	count = 0
	while represented_by != represented_by_old and count < 10:
		count += 1
		point_count_for_mean = [0 for i in range(0,k)]
		represented_by_old = represented_by
		represented_by = [worst_mean for i in range(0,blob_X.shape[0])]
		for i in range(0,blob_X.shape[0]):
			best_dist = INF
			best_mean = worst_mean
			for j in range(0,k):
				negated_blob = np.array([-blob_X[i][0],-blob_X[i][1]])
				z = RBF(blob_X[i],mean[j]) + RBF(negated_blob,mean[j])
				dist = z
				if dist < best_dist:
					best_dist = dist
					best_mean = j
					represented_by[i] = j
		mean0 = np.array([-INF,-INF])
		metric0 = INF
		mean1 = np.array([-INF,-INF])
		metric1 = INF
		for i in range(0,blob_X.shape[0]):
			metric = 0
			if represented_by[i] == 0:
				for j in range(0,blob_X.shape[0]):
					if represented_by[j] == 0:
						negated_blob = np.array([-blob_X[i][0],-blob_X[i][1]])
						metric += RBF(blob_X[i],blob_X[j]) + RBF(negated_blob,blob_X[j])
				if metric < metric0:
					metric0 = metric
					mean0 = blob_X[i]
					mean[0] = mean0
					
			elif represented_by[i] == 1:
				for j in range(0,blob_X.shape[0]):
					if represented_by[j] == 1:
						negated_blob = np.array([-blob_X[i][0],-blob_X[i][1]])
						metric += RBF(blob_X[i],blob_X[j]) + RBF(negated_blob,blob_X[j])
				if metric < metric1:
					metric1 = metric
					mean1 = blob_X[i]
					mean[1] = mean1
				
	plt.scatter(blob_X[:,0], blob_X[:,1],c = represented_by)
	plt.plot(mean[0][0],mean[0][1])
	plt.plot(mean[1][0],mean[1][1])
	plt.show()



def genMean():
	return np.array([random.uniform(-1.5,1.5),random.uniform(-0.5,2)],dtype = float)
	#return np.random.rand(2)

def genVariance():
	x = np.abs(np.random.rand())
	y = np.random.rand()
	z = np.abs(np.random.rand())
	return np.array([[x,y],[y,z]])
	
def pdf(x,m,v):
	den = np.power(2*np.pi*np.abs(np.linalg.det(v)),-1)
	z = np.dot(np.dot((x-m).transpose(),np.linalg.pinv(v)),(x-m))
	num = np.exp(-z)
	return (num*den)[0][0]

def genPrior():
	x = random.uniform(0,1)
	y = random.uniform(0,1)
	z = random.uniform(0,1)
	add = x+y+z
	x = float(x)/add
	y = float(y)/add
	z = float(z)/add
	return [x,y,z]




def EM(data):
	superList = []
	bestLikelihood = -9999999
	best_r = []
	best_m = []
	best_v = []
	for times in range(0,5):
		m = np.array([genMean() for i in range(3)],dtype = float)
		v = np.array([genVariance() for i in range(3)],dtype = float)
		pri = np.array(genPrior(),dtype = float)
		r = np.array([[0 for i in range(3)] for j in range(len(data))],dtype = float)
		r_numer = np.array([[0 for i in range(3)] for j in range(len(data))],dtype = float)
		likelihoodList = []
		for iteration in range(0,20):
			for i in range(len(data)):
				for k in range(3):
					r_numer[i][k] = pdf(data[i].reshape(2,1),m[k].reshape(2,1),v[k]) * pri[k]
			for i in range(len(data)):
				for k in range(3):
					r[i][k] = r_numer[i][k]/(r_numer[i][0]+r_numer[i][1]+r_numer[i][2])

			r_add = np.array([0.0 for i in range(3)],dtype = float)
			r_total = 0
			for i in range(len(data)):
				for k in range(3):
					r_add[k] += r[i][k]
					r_total += r[i][k]
			#print r_add
			for k in range(3):
				pri[k] = r_add[k]/r_total

			for k in range(3):
				temp_m = np.array([0,0],dtype = float)
				temp_m.reshape(2,1)
				for i in range(len(data)):
					temp_m[0] += (r[i][k]*data[i][0]/r_add[k])
					temp_m[1] += (r[i][k]*data[i][1]/r_add[k])
				m[k] = temp_m
			#print m

			for k in range(3):
				temp_v = np.array([[0,0],[0,0]],dtype = float)
				temp_v.reshape(2,2)
				for i in range(len(data)):
					b = data[i] - m[k]
					temp_v[0][0] += r[i][k]*b[0]*b[0]/r_add[k]
					temp_v[1][0] += r[i][k]*b[0]*b[1]/r_add[k]
					temp_v[0][1] += r[i][k]*b[0]*b[1]/r_add[k]
					temp_v[1][1] += r[i][k]*b[1]*b[1]/r_add[k]
				v[k] = temp_v
			#print v
			likelihood = 0
			for i in range(len(data)):
				#print r_numer[i][0]+r_numer[i][1]+r_numer[i][2]
				likelihood += math.log(r_numer[i][0]+r_numer[i][1]+r_numer[i][2])
			if iteration != 0 :
				likelihoodList.append(likelihood)
				if likelihood > bestLikelihood:
					bestLikelihood = likelihood
					best_r = r
					best_m = m
					best_v = v
		superList.append(likelihoodList)
	for i in superList:
		plt.plot(i)
	plt.show()

	print "variance:",v
	print "mean:",m	
	
	colors = []
	for row in best_r:
		colors.append(np.argmax(row))
	#print colors
	
	plt.scatter(data[:,0],data[:,1],c=colors)	
	plt.ylabel('loglikelihood')
	plt.xlabel('iterations')
	plt.show()






def main():
	blob_X = genfromtxt('hw5_blob.csv', delimiter=',')
	circle_X = genfromtxt('hw5_circle.csv', delimiter=',')
	print "Kmeans clustering on blob:"	
	kmeans(blob_X)
	print "Kmeans clustering on circle:"
	kmeans(circle_X)
	print "kernel Kmeans on circle:"
	kernelkmeans(circle_X)
	print "EM on blob:"
	EM(blob_X)

main()
