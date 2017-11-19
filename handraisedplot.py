import matplotlib.pyplot as plt
from random import *

handRaisedArray = [int]*100

for i in range(100):
    handRaisedArray[i] = int(random()*100%2)
#generating an empty array of 1's and 0's. In final product should obviously be populated with array of whether hands are raised

percentRaised = 5
temp = 0
for i in range(100):
    temp = temp + handRaisedArray[i]

percentRaised = temp/100
#this determines what percentage of the array is 1 (hand raised)
#calculated for plotting purposes


print(temp)
print(handRaisedArray)
print("hand raised %s percent of the time  "   %(percentRaised*100))
plt.plot(handRaisedArray)
#debatable way to show if hands are raised. In this example it looks shitty b/c
#the 1's and 0's alternate a lot. Wouldn't happen if the changes are less frequent, which is ideally how the software would work

plt.ylabel('Hand Raised?')
plt.xlabel('Time')
plt.show()



