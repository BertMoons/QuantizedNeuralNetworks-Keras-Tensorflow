# test using cpu only
cpu = False

# type of network to be trained, can be bnn, full-bnn, qnn, full-qnn, tnn, full-tnn
network_type = 'full-qnn'
# bits can be None, 2, 4, 8 , whatever
bits=None
wbits = 4
abits = 4
# finetune an be false or true
finetune = False

dataset='MNIST'
dim=28
channels=1
classes=10

#regularization
kernel_regularizer=0.
activity_regularizer=0.

# width and depth
nla=2
nfa=64
nlb=3
nfb=128
nlc=3
nfc=128

#learning rate decay, factor => LR *= factor
decay_at_epoch = [0, 25, 80 ]
factor_at_epoch = [1, .1, 1]
kernel_lr_multiplier = 10

# debug and logging
progress_logging = 1 # can be 0 = no std logging, 1 = progress bar logging, 2 = one log line per epoch
epochs = 100
batch_size = 64
lr = 0.001
decay = 0.000025


# important paths
out_wght_path = './weights/{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(dataset,network_type,abits,wbits,nla,nfa,nlb,nfb,nlc,nfc)
tensorboard_name = '{}_{}_{}b_{}b_{}_{}_{}_{}_{}_{}.hdf5'.format(dataset,network_type,abits,wbits,nla,nfa,nlb,nfb,nlc,nfc)
