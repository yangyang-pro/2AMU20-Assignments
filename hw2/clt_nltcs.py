from binary_clt import BinaryCLT
import numpy as np
import time
import csv


def load(path):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        data = np.array(list(reader)).astype(np.float)
    return data


nltcs_train = load('./nltcs.train.data')
nltcs_test = load('./nltcs.test.data')
nltcs_marginals = load('./nltcs_marginals.data')

clt = BinaryCLT(data=nltcs_train, root=0, alpha=0.01)

# task 2e.1
print("Tree: ", clt.get_tree())

# task 2e.2
print("Log params: ", clt.get_log_params())

# task 2e.3
ll_train = clt.log_prob(nltcs_train)
ll_test = clt.log_prob(nltcs_test)
print('Average Train LL: ', np.mean(ll_train))
print('Average Test  LL: ', np.mean(ll_test))

# task 2e.4
start_exhaustive = time.time()
ll_mar_exhaustive = clt.log_prob(nltcs_marginals, exhaustive=True)
end_exhaustive = time.time()
start_messpass = time.time()
ll_mar_messpass = clt.log_prob(nltcs_marginals, exhaustive=False)
end_messpass = time.time()
print('Average marginals LL (exhaustive): ', np.mean(ll_mar_exhaustive))
print('Average marginals LL (mess.pass.): ', np.mean(ll_mar_messpass))
print('Are LLs equal?: ', (np.abs(ll_mar_exhaustive - ll_mar_messpass) < 1e-10).all())

# task2e.5
print('Time exhaustive inference: ', end_exhaustive - start_exhaustive) # about 7 minutes using the second technique
print('Time mess.pass. inference: ', end_messpass - start_messpass)

# task 2e.6
for _ in range(10):
    samples = clt.sample(1000)
    ll_samples = clt.log_prob(samples)
    print('Average samples LL: ', np.mean(ll_samples))
