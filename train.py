from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.losses import squared_hinge
import os
import argparse
import keras.backend as K

from models.model_factory import build_model
from utils.config_utils import Config
from utils.load_data import load_dataset

# parse arguments
parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('-c', '--config_path', type=str,
                default=None, help='Configuration file')
parser.add_argument('-o' ,'--override',action='store',nargs='*',default=[])

arguments = parser.parse_args()
override_dir = {}

for s in arguments.override:
    s_s = s.split("=")
    k = s_s[0].strip()
    v = "=".join(s_s[1:]).strip()
    override_dir[k]=v
arguments.override = override_dir


cfg = arguments.config_path
cf = Config(cfg, cmd_args = arguments.override)


# if necessary, only use the CPU for debugging
if cf.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ## Construct the network
print('Construct the Network\n')

# In[4]:
model = build_model(cf)



print('setting up the network and creating callbacks\n')

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
checkpoint = ModelCheckpoint(cf.out_wght_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
tensorboard = TensorBoard(log_dir='./logs/' + str(cf.tensorboard_name), histogram_freq=0, write_graph=True, write_images=False)

print('loading data\n')

train_data, val_data, test_data = load_dataset(cf.dataset)

# learning rate schedule
def scheduler(epoch):
    if epoch in cf.decay_at_epoch:
        index = cf.decay_at_epoch.index(epoch)
        factor = cf.factor_at_epoch[index]
        lr = K.get_value(model.optimizer.lr)
        IT = train_data.X.shape[0]/cf.batch_size
        current_lr = lr * (1./(1.+cf.decay*epoch*IT))
        K.set_value(model.optimizer.lr,current_lr*factor)
        print('\nEpoch {} updates LR: LR = LR * {} = {}\n'.format(epoch+1,factor, K.get_value(model.optimizer.lr)))
    return K.get_value(model.optimizer.lr)
    
lr_decay = LearningRateScheduler(scheduler)


#sgd = SGD(lr=cf.lr, decay=cf.decay, momentum=0.9, nesterov=True)
adam= Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=cf.decay)

print('compiling the network\n')
model.compile(loss=squared_hinge, optimizer=adam, metrics=['accuracy'])

if cf.finetune:
    print('Load previous weights\n')
    model.load_weights(cf.out_wght_path)
else:
    print('No weights preloaded, training from scratch\n')

print('(re)training the network\n')
model.fit(train_data.X,train_data.y,
            batch_size = cf.batch_size,
            epochs = cf.epochs,
            verbose = cf.progress_logging,
            callbacks = [checkpoint, tensorboard,lr_decay],
            validation_data = (val_data.X,val_data.y))


print('Done\n')



