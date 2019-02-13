import numpy as np
import matplotlib.pyplot as plt
import math
a=3
b=5
e=(np.sqrt(1-((a**2)/float(b**2))))
C=np.array([0,0])
F1=np.array([0,b*e])
F2=np.array([0,-b*e])
print(e)
print(F1)
print(F2)
A = np.array([1,0])
B = np.array([0,1])
P= np.array([3/2.0,5*(np.sqrt(3))/2.0])
Q = np.array([6,0])

t = np.linspace(-np.pi,np.pi,1000)
k=np.linspace(-1,1,10)
x_PQ=np.zeros((2,10))
x_R1F1=np.zeros((2,10))
x_R2F2=np.zeros((2,10))
x_AB = np.zeros((2,1000))
for j in range(10):
	temp1 = P + k[j]*(Q-P)
	x_PQ[:,j]=temp1.T
for i in range(1000):
	l = a*(np.cos(t[i]))
	m = b*(np.sin(t[i]))
	x_AB[:,i] = (l) * A + (m) * B
PQ=np.vstack((P,Q)).T
dvec=np.array([1,-1])
omat=np.array([[0,1],[-1,0]])
def norm_vec(PQ):
	return np.matmul(omat,np.matmul(PQ,dvec))

n1=norm_vec(PQ)
s1=np.sqrt((n1[0]**2)+(n1[1]**2))
s2=(n1[0]**2)+(n1[1]**2)
p=np.matmul(n1,P)
q=np.matmul(n1,F1)
r=np.matmul(n1,F2)
R1=np.array([F1[0]-((n1[0]*(q-p))/float(s2)),F1[1]-((n1[1]*(q-p))/float(s2))])
R2=np.array([F2[0]-((n1[0]*(r-p))/float(s2)),F2[1]-((n1[1]*(r-p))/float(s2))])
print(R2)
d1=(p-q)/float(s1)
d2=(p-r)/float(s1)
print(d1,d2)
print((d1*d2))
print(a**2)
w=np.linspace(0,1,10)
for u in range(10):
	temp2 = F1 + w[u]*(R1-F1)
	x_R1F1[:,u]=temp2.T
	temp3 = F2 + w[u]*(R2-F2)
	x_R2F2[:,u]=temp3.T

plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_R1F1[0,:],x_R1F1[1,:],label='$perpendicular 1$')
plt.plot(x_R2F2[0,:],x_R2F2[1,:],label='$perpendicular 2$')
plt.plot(F1[0],F1[1],'o')
plt.text(F1[0]*(1+0.1),F1[1]*(1-0.1),'F1')
plt.plot(F2[0],F2[1],'o')
plt.text(F2[0]*(1+0.1),F2[1]*(1-0.1),'F2')
plt.plot(P[0],P[1],'o')
plt.text(P[0]*(1+0.1),P[1]*(1-0.1),'P')
plt.plot(Q[0],Q[1],'o')
plt.text(Q[0]*(1-0.03),Q[1]*(1-0.05),'Q')
plt.plot(R1[0],R1[1],'o')
plt.text(R1[0]*(1-0.1),R1[1]*(1-0.1),'R1')
plt.plot(R2[0],R2[1],'o')
plt.text(R2[0]*(1-0.03),R2[1]*(1-0.05),'R2')
plt.plot(C[0],C[1],'o')
plt.text(C[0]*(1+0.1),C[1]*(1-0.1),'C')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.show()
