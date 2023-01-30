#the purpose of this file is to solve the Brachistrone problem per ME575 hw1 problem 2.
#Joseph Carter
#Version 1.0

#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

#function definition
def f(y,n,start,end,mu_k):
    #these can be inferred from the problem
    h = start[1] #h is defined as the starting y location of the ball
    x = np.linspace(start[0],end[0],n) #discretization of x
    y = np.insert(y,0,start[1]) #regardless of the prediction, y must start at its fixed startpoint
    y = np.append(y,end[1]) #regardless of the prediction, y must end at its fixed endpoint

    #formula for time
    f = 0 #initialize time at 0
    for i in range(0,n-1):
        deltax = x[i+1] - x[i] #shorthand writing of deltax
        deltay = y[i+1] - y[i] #shorthand writing of deltay
        f += np.sqrt(2 / 9.8) * (np.sqrt(deltax**2 + deltay**2)) / (np.sqrt(h - y[i+1] - mu_k * x[i+1]) + np.sqrt(h - y[i] - mu_k * x[i])) #this formula was copied from example D.1.7
    return f

# provided information from the problem
start = [0,1] #the ball starts at x = 0, y = 1
end = [1,0] #the ball ends at x = 1, y = 0
n = 12 #there are a total of 12 nodes from start to end, with the endpoints being fixed. This leaves 10 design variables
mu_k = 0.3 #kinetic friction coefficient is 0.3

#make initial guess


#calculate y_optimized
p = 0
samples = [4, 8, 16, 32, 64, 128]
time_elapsed = np.zeros([len(samples),1])
for n in samples:
    y0 = np.zeros([n-2,1]) #I make the initial guess of all zeros for convenience
    start_time = time.time()
    ans = minimize(f,y0,args=(n,start,end,mu_k)) #utilization of the function written above
    end_time = time.time()
    time_elapsed[p] = end_time - start_time
    p = p + 1

design_variables = np.linspace(12,130,len(time_elapsed))

plt.figure()
plt.plot(design_variables,time_elapsed)
plt.xlabel('Number of Design Variables')
plt.ylabel('Time (s)')
plt.title('Dimensionality v Time')
plt.show()

# y_optimized = ans.x #optimal path is designated as x in the 'ans' dict
# time = ans.fun #elapsed time is designated as fun in the 'ans' dict
# y_optimized = np.insert(y_optimized,0,1) #appends start position
# y_optimized = np.append(y_optimized,0) #appends end position

# #plot the path
# x_plot = np.linspace(0,1,12) #creates x_plot variable for plotting
# plt.figure() 
# plt.plot(x_plot,y_optimized)
# plt.xlabel('Position x (m)')
# plt.ylabel('Position y (m)')
# plt.title('Brachistochrone Problem Optimal Path (n = 12)')
# plt.show()

# print(time)

