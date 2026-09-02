#!/usr/bin/env python

import math, sys
import numpy as np
import matplotlib.pyplot as plt



d = {}
files = sys.argv[1:]


class GraphAttributes:
    def __init__(self):
        # these are defaults for the graphs
        self.markersize = 8
        self.alpha = 1.0
        self.label = ''
        self.mec = 'k'
        self.linestyle = '-'

ga = {}
for i in range(5):
    g = GraphAttributes()
    if i == 0:
        g.label = "ST/2"
        g.linestyle = "ro--"
        g.mec = 'r'
    elif i == 1:
        g.label = "ST"
        g.linestyle = "bD-."
    elif i == 2:
        g.label = "M = 2M$_{sun}$"
        g.linestyle = "mv-"
        g.mec = 'm'
    elif i == 3:
        g.label = "M = 4M$_{sun}$"
        g.linestyle = "cp--"
    else:
        g.label = "STx3"
        g.linestyle = 'ks-'
    ga[i] = g
ga['final'] = ga[4]
ga['alone'] = GraphAttributes()
ga['alone'].label = 'Model'

#print 'ga', ga
#print 'ga.keys()', ga.keys()
#for k, v in ga.iteritems():
#    print k, v.label
#sys.exit(1)

# 'ro--', mec = 'w', markersize=10, alpha=1.0, label = "1st TP"
# 'ks-', markersize=10, alpha=1.0, label = "Final TP"

def graphAttrs(i):
    return ga[i]

class StuffInFile:
    pass

def readfile(fn):
    # Read file 1
    f = open(fn, 'r')

    # Read and ignore header lines
    header1 = f.readline()
    header2 = f.readline()

    # read the whole file into a single variable, which is a list of every row of the file.
    lines = f.readlines()
    f.close()

    o = StuffInFile()

    # initialize: 
    o.name = []; o.z = []; o.loge = []
    o.xh = []; o.xfe = []; o.xo = []
    o.x = []
    
    # scan the rows of the file stored in lines, and put the values into some variables:
    for line in lines:
        p = line.split()
        o.name.append((p[0])) #no float or int needed? might break!
        o.z.append(int(p[1])) 
        o.loge.append(float(p[2]))
        o.xh.append(float(p[3]))
        o.xfe.append(float(p[4]))
        o.xo.append(float(p[5]))
        o.x.append(float(p[6]))

    o.x1 = np.array(o.z)
    o.y1 = np.array(o.loge)
    o.y2 = np.array(o.xh)
    o.y3 = np.array(o.xfe)
    o.y4 = np.array(o.xo)
    o.y5 = np.array(o.x)
    
    return o

i = 0
for f in files:
    d[f] = readfile(f)
    d[f].g = graphAttrs(i)
    i += 1
print 'files', files, 'len(files)', len(files), 'files[-1]', files[-1]
if len(files) == 1:
    d[files[0]].g = graphAttrs('alone')
else:
    d[files[-1]].g = graphAttrs('final')

#plot(sdfks, label=o.g.label, linestyle=o.g.linestyle, ... )

#for f in files:
#    print f, d[f], d[f].name, d[f].g.label

#sys.exit(1)    

plt.rc('lines', linewidth=2.0)

fig1 = plt.figure(figsize=(8, 8))

ax1 = plt.subplot(211)
for f in files:
    o = d[f]
    plt.plot(o.x1, o.y3, o.g.linestyle, mec = o.g.mec, markersize=o.g.markersize, alpha=o.g.alpha, label = o.g.label)
plt.ylabel('[X/Fe]',fontsize=16, labelpad=15)
plt.ylim(-1.25,4.0)

# Now add the legend with some customizations.
legend = ax1.legend(loc='upper right', shadow=True, numpoints = 1, fancybox=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('small')

for label in legend.get_lines():
    label.set_linewidth(2.0)  # the legend line width

plt.minorticks_on()

plt.xlabel('Proton number, Z',fontsize=16, labelpad=20)
plt.xticks(color='k', size=16)
plt.xticks(np.arange(1.0, 90., 5.0))
plt.xlim(5.0,26.0)

ax2 = plt.subplot(212)
for f in files:
    o = d[f]
    plt.plot(o.x1, o.y3, o.g.linestyle, mec = o.g.mec, markersize=o.g.markersize, alpha=o.g.alpha, label = o.g.label)
#plt.legend(loc="upper right", numpoints=1, shadow=True, fancybox=True)
#plt.ylabel('log $\epsilon$(x)',fontsize=16, labelpad=15)
plt.ylabel('[X/Fe]',fontsize=16, labelpad=15)
#plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim(-0.5,3.5)

plt.minorticks_on()

plt.xlabel('Proton number, Z',fontsize=16, labelpad=20)
plt.xticks(color='k', size=16)
plt.xticks(np.arange(20., 90., 5.0))
plt.xlim(26.0,85.0)

# Now add the legend with some customizations.
legend = ax2.legend(loc='upper left', shadow=True, numpoints = 1, fancybox=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('small')

for label in legend.get_lines():
    label.set_linewidth(2.0)  # the legend line width

plt.minorticks_on()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig1.savefig('fig1.pdf')
plt.show()

