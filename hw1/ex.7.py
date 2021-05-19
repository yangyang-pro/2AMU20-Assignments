import itertools
import random
import scipy.stats as sct
import numpy as np
import matplotlib.pyplot as plt

# Given the positive test result, what is the probability that that you actually have the disease?

#simulations 
N = 1000000
#prob. of indicate positive given positive p(Y= 1|X= 1) 
pp = 0.99
#prob. of indicate negative given negative p(Y= 0|X= 0) 
nn = 0.99
#prob. of someone being positive p(X= 1)
pos = 0.0001
#prob. of someone being negative p(X= 0) 
neg = 0.9999


#Counter for the event when a person is diseased and the test indicates positive
count_P_test_P = 0

#Counter for the event of the polygraph indicating that the applicant is negative
test_P = 0


for n in range(N):
    #Simulate if a person is positive or not
    disease = np.random.choice(["yes", "no"], size = 1, p = [pos, neg])[0]

    if disease == "yes":
        test = np.random.choice(["disease", "no disease"], size = 1, p = [pp, 1 - pp])[0]

        if test == "disease":
            count_P_test_P = count_P_test_P + 1
            test_P = test_P + 1
    else:
        #Given that the person is diseased
        #Simulate a decision of the test
        #according to the given probabilities
        test = np.random.choice(["disease1", "no disease1"], size = 1, p = [pp, 1 - pp])[0]

        if test == "disease1":
            test_P = test_P + 1

#approxiamte the probability
prob_approx = count_P_test_P / test_P

print(f'The approximate probability is {prob_approx}')