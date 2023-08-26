import matplotlib.pyplot as plt
from scipy import stats

GPU = open("GPU.txt", "r")

x_points = []
y_points = []
for line in GPU:
    line = line.split(": ")
    x_points.append(float(line[0]))
    y_points.append(float(line[1]))

slope, intercept, r, p, std_err = stats.linregress(x_points, y_points)
def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x_points))
plt.plot(x_points, y_points, ":", color='#FF6e6e')
plt.plot(x_points, mymodel, color='#FF0000', label='Parallel Reduction')

plt.title("time for eval of test_data as a function of size of hidden layer diagram")
plt.xlabel('neurons in the hidden layer')
plt.ylabel('time in ms')
plt.legend()

plt.savefig("plot.png")