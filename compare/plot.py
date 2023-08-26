import matplotlib.pyplot as plt
from scipy import stats

def plot_graph(filename, label, color1, color2):

    file = open(filename, "r")

    x_points = []
    y_points = []
    for line in file:
        line = line.split(": ")
        x_points.append(float(line[0]))
        y_points.append(float(line[1]))

    slope, intercept, r, p, std_err = stats.linregress(x_points, y_points)
    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x_points))
    #plt.plot(x_points, y_points, ":", color=color2)
    plt.scatter(x_points, y_points, 1, color=color2)
    plt.plot(x_points, mymodel, color=color1, label=label)

plot_graph("GPU.txt", "Parallel Reduction", "#FF0000", "#FF6e6e")
plot_graph("CPU.txt", "CPU adding", "#2d8108", "#7ebc63")


plt.title("time for eval of test_data as a function of size of hidden layer diagram")
plt.xlabel('neurons in the hidden layer')
plt.ylabel('time in ms')
plt.legend()

plt.savefig("plot.png")