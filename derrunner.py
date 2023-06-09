import subprocess
import math

base = 1.5

#fcw, fcb, cnw, cnb, l2, mom
params = [math.log(0.008) / math.log(base), math.log(0.1) / math.log(base), math.log(0.1) / math.log(base), math.log(0.1) / math.log(base), -1000000, -100000]
names = ["fcw", "fcb", "cnw", "cnb", "l2", "mom"]

fac = 10
q = 1.5

s = 7
st = [10, 3, 3]

testfile = './fully_connected'

def evaluate():
    process = subprocess.Popen([testfile] + [str(base**a) for a in params], stdout=subprocess.PIPE)

    output, _ = process.communicate()

    output = output.decode()
    return float(output)

dp = {}

def score():
    if str(params) in dp:
        return dp[str(params)]
    runs = 3
    average = 0
    for _ in range(runs):
        average += evaluate()/runs
    dp[str(params)] = average
    return dp[str(params)]

def betterrange(l, r, s):
    li = []
    while l < r:
        li.append(l)
        l += s
    return li

def searchopt(i):
    l = params[i] - s
    r = params[i] + s
    for ste in st[1:]:
        ttest = betterrange(l, r + 10**-6, (r-l)/ste)
        vals = []
        for a in ttest:
            params[i] = a
            print("testing {}={}\n============================================".format(names[i], base**params[i]))
            vals.append(score())
            print("============================================")
        start = -1
        best = 0
        for test in range(len(vals) - 1):
            if vals[test] + vals[test+1] > best:
                start = test
                best = vals[test]
        l = ttest[start]
        r = ttest[start+1]
    params[i] = (l + r) / 2

searchopt(1)
searchopt(0)

testfile = "./cnn"

searchopt(3)
searchopt(2)