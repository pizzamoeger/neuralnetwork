import matplotlib.pyplot as plt
from scipy import stats

def plot_graph(filename, label, color1, color2):

    file = open(filename, "r")

    x_points = []
    y_points = []
    counter = 0
    for line in file:
        if (counter > 176):
            break
        line = line.split(": ")
        x_points.append(float(line[0]))
        y_points.append(float(line[1]))
        counter += 1

    slope, intercept, r, p, std_err = stats.linregress(x_points, y_points)
    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x_points))
    #plt.plot(x_points, y_points, ":", color=color2)
    #plt.scatter(x_points, y_points, 10, color=color2)
    plt.plot(x_points, y_points, color=color1, label=label)
    #plt.plot(x_points, mymodel, color=color1, label=label)

plot_graph("GPU.txt", "Parallel Reduction", "#FF0000", "#FF6e6e")
#plot_graph("CPU.txt", "CPU adding", "#2d8108", "#7ebc63")
plot_graph("GPU_double.txt", "Parallel Reduction using doubles", "#0b5394", "#6fa8dc")
#plot_graph("GPU_AtmoicAdd.txt", "AtomicAdd", "#f1c232", "#ffe599")


plt.title("time for eval of test_data as a function of size of hidden layer diagram")
plt.xlabel('neurons in the hidden layer')
plt.ylabel('time in ms')
plt.legend()

plt.savefig("plot.png")