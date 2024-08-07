import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'..\\Data\\FinalData\\RadiationShielding.csv', delimiter=',', header=0,
                   names=['NumSim','m','q','I','a','LoopMass','CelerityCrit','InitialCelerity',
                          'ClosestApproach','ClosestUnc','CvT'])

data['NormCelerity'] = data['InitialCelerity']/data['CelerityCrit']
data['NormR'] = data['ClosestApproach']/data['a']
data['NormRUnc'] = data['ClosestUnc']/data['a']

data = data.sort_values(by="NormCelerity")

ControlCelerity = []
ControlR = []
ControlRUnc = []
TestCelerity = []
TestR = []
TestRUnc = []
ArtifactCelerity = []
ArtifactR = []
ArtifactRUnc = []
i = 0
while i < len(data):
    if data.loc[i,'CvT'] == 'Control':
       ControlCelerity.append(data.loc[i,'NormCelerity'])
       ControlR.append(data.loc[i,'NormR'])
       ControlRUnc.append(data.loc[i,'NormRUnc'])
    elif data.loc[i,'CvT'] == 'Test' and data.loc[i,'NormCelerity'] < 1:
       TestCelerity.append(data.loc[i,'NormCelerity'])
       TestR.append(data.loc[i,'NormR'])
       TestRUnc.append(data.loc[i,'NormRUnc'])
    else:
       ArtifactCelerity.append(data.loc[i,'NormCelerity'])
       ArtifactR.append(data.loc[i,'NormR'])
       ArtifactRUnc.append(data.loc[i,'NormRUnc'])
    i+=1

a,b,c = np.polyfit(TestCelerity, TestR, 2)

def func(x, a, b, c, d):
   y = a*np.log(b*np.arctan(c*x)) + d
   return y

p, p_cov = curve_fit(func, TestCelerity, TestR)

x = np.linspace(0.001, 1000, 1000000)
y = p[0]*np.log(p[1]*np.arctan(p[2]*x)) + p[3]
print(p)

plt.plot(x,y)
#plt.errorbar(ControlCelerity, ControlR, yerr=ControlRUnc, xerr=None, fmt='d', color='blue')
plt.errorbar(ArtifactCelerity, ArtifactR, yerr=ArtifactRUnc, xerr=None, fmt='s', color='red')
plt.errorbar(TestCelerity, TestR, yerr=TestRUnc, xerr=None, fmt='o', color='green')
plt.axvline(x=1)
plt.ylabel('Distance of Closest Approach/Loop Radius')
plt.xlabel('Initial Celerity/Critical Celerity')
plt.xscale('log')
plt.yscale('log')
plt.title('Radiation Deflection vs. Normalized Proton Celerity')
plt.xlim(10**-3, 10**3)
plt.ylim(bottom=10**-3)

plt.show()
