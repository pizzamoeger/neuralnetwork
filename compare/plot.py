import matplotlib.pyplot as plt
from scipy import stats

def plot_graph(filename, label, color1, color2, STOP):

    file = open(filename, "r")

    x_points = []
    y_points = []
    for line in file:
        line = line.split(": ")
        if (float(line[0]) > STOP):
            continue
        x_points.append(float(line[0]))
        y_points.append(float(line[1]))

    slope, intercept, r, p, std_err = stats.linregress(x_points, y_points)
    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x_points))
    #plt.plot(x_points, y_points, ":", color=color2, linewidth = '3')
    #plt.scatter(x_points, y_points, s=10, color=color2)
    plt.plot(x_points, y_points, color=color1, label=label)
    #plt.plot(x_points, mymodel, color=color1, label=label)

#plt.ylim([0, 1500])


#plot_graph("GPU.txt", "Parallel Reduction (floats)", "#FF0000", "#FF6e6e", 300)
#plot_graph("GPU_double.txt", "Parallel Reduction (doubles)", "#0b5394", "#6fa8dc", 300)
#plot_graph("CPU.txt", "CPU adding (floats) one hidden layer", "#2d8108", "#7ebc63", 110)
#plot_graph("GPU_AtmoicAdd.txt", "AtomicAdd (floats)", "#f1c232", "#ffe599", 110)

plot_graph("GPU_n_squared.txt", "Parallel Reduction (doubles)", "#6fa8dc", "#6fa8dc", 512)
plot_graph("CPU_n_squared.txt", "CPU adding (floats) two hidden layers", "#7ebc63", "#7ebc63", 512)

plt.title("time for eval of test_data as a function of size of hidden layer diagram")
plt.xlabel('neurons in the hidden layer')
plt.ylabel('time in ms')
plt.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,-0.3), ncols=2)

plt.savefig("plot.svg", bbox_inches="tight")