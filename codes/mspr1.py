import sys
import pandas as pd
import numpy as np
#from google.colab import drive 
#!gdown --id '1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm' --output data.zip
#!unzip data.zip
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('./train.csv', encoding = 'big5')

#Model parameters

dayhours = 9 #Nearest i hours taken
data_entries_per_month = 480 - dayhours
reject_first_n_hours= 0 #Just for convenience, nearest (i-n) hours taken, 
                        #same no of data entries as the configuration with the nearest i hours taken
assert reject_first_n_hours < dayhours and reject_first_n_hours >=0 
only_pm2_5 = False # Use only the pm2.5 attribute as the prediction data
lr = 1 # learning rate
noitrs = 1000 #no of iterations
generateImage=True # generate image for performance under different lr(1,10,100,1000), itrs = 1000
image_path='./different_lrs' # image path



data = data.iloc[:, 3:]  #slice objects, [all rows, columns starting from index 3] (slice objects)
data[data == 'NR'] = 0 # data values with nr is set to 0 
raw_data = data.to_numpy() # convert the data array into an numpy array


month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample


x = np.empty([12 * data_entries_per_month, (1 if only_pm2_5 else 18) * (dayhours-reject_first_n_hours)], dtype = float)
y = np.empty([12 * data_entries_per_month, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 23 - dayhours:
                continue
            if only_pm2_5:
                x[month * data_entries_per_month + day * 24 + hour, :] = month_data[month][ 8,day * 24 + hour + reject_first_n_hours : day * 24 + hour + dayhours].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            else:
                x[month * data_entries_per_month + day * 24 + hour, :] = month_data[month][ :,day * 24 + hour + reject_first_n_hours : day * 24 + hour + dayhours].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * data_entries_per_month + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + dayhours] #value , 10th (index 9) row is the PM2.5 value
print(x)
print(y)

mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
x

x_original = np.copy(x)
y_origianl = np.copy(y)

import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))


# Modified version
#Linear regression 
x = x_train_set # using part of the data(training set)
total_data_entries = np.shape(x)[0]
y = y_train_set

dimensions = (1 if only_pm2_5 else 18)* (dayhours - reject_first_n_hours) + 1  # another dimension for the 'b' value
w= np.ones([dimensions,1], dtype=np.float128) #column vector
x = np.concatenate((np.ones([total_data_entries, 1]), x), axis = 1).astype(np.float128) #create a column vector of [1,1,1,1,...], appends it to the left of the data matrix x


adagrad = np.zeros([dimensions, 1])
eps = 0.0000000001

for i in range(noitrs):
    yprime = np.dot(x,w)
    #loss = np.sum(np.power(yprime - y,2))
    loss = np.sqrt(np.sum(np.power(yprime - y, 2))/total_data_entries) 

    gradient = 2* np.dot(x.transpose(),yprime-y)

    adagrad += gradient ** 2
    w = w - lr * gradient/np.sqrt(adagrad + eps)
    #print('finished itr %d' % i)

np.save('weight.npy', w)

#Validation set
w = np.load('weight.npy')
x_validation = np.concatenate((np.ones([np.shape(x_validation)[0], 1]), x_validation), axis = 1).astype(np.float128)
validate_y = np.dot(x_validation, w)
loss = np.sqrt(np.sum(np.power(y_validation - validate_y, 2))/total_data_entries)
#print('dayhours: %d - %d, loss %f' % (reject_first_n_hours,dayhours, loss))
#testing
#./test.csv can be used only if dayhours = 9
if len(sys.argv) > 1:
    input_route = sys.argv[1]
else:
    input_route = './test.csv'
print('Reading from: %s', input_route)
testdata = pd.read_csv(input_route, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, (1 if only_pm2_5 else 18)*(dayhours-reject_first_n_hours)], dtype = float)
if only_pm2_5:
    for i in range(240):
        test_x[i, :] = test_data[18 * i+ 8, reject_first_n_hours:].reshape(1, -1)
else:
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18* (i + 1), reject_first_n_hours:].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x

#generating results
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
ans_y

if len(sys.argv) > 2:
    output_route = sys.argv[2]
else:
    output_route = 'submit.csv'

print('Outputing to: %s', output_route)
import csv
with open(output_route, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

print('dayhours: %d - %d, loss %f' % (reject_first_n_hours,dayhours, loss))

if generateImage:
    print('Generating the image...')
    #Linear regression
    #following block is for generating data and plot for different lr rates
    #lr = 1
    lr = 1

    x = np.copy(x_original)

    x = x_train_set # using part of the data(training set)
    total_data_entries = np.shape(x)[0]
    y = y_train_set

    dimensions = (1 if only_pm2_5 else 18)* (dayhours - reject_first_n_hours) + 1  # another dimension for the 'b' value
    w= np.ones([dimensions,1], dtype=np.float128) #column vector
    x = np.concatenate((np.ones([total_data_entries, 1]), x), axis = 1).astype(np.float128) #create a column vector of [1,1,1,1,...], appends it to the left of the data matrix x


    adagrad = np.zeros([dimensions, 1])
    eps = 0.0000000001

    lr1 = np.empty(noitrs)

    for i in range(noitrs):
        yprime = np.dot(x,w)
        #loss = np.sum(np.power(yprime - y,2))
        loss = np.sqrt(np.sum(np.power(yprime - y, 2))/total_data_entries) 

        lr1[i] = loss
        
        gradient = 2* np.dot(x.transpose(),yprime-y)
        adagrad += gradient ** 2
        w = w - lr * gradient/np.sqrt(adagrad + eps)
        #print('finished itr %d' % i)

    print('Finish collecting data for lr = %f'% lr)

    #lr = 10
    lr = 10

    x = np.copy(x_original)

    x = x_train_set # using part of the data(training set)
    total_data_entries = np.shape(x)[0]
    y = y_train_set

    dimensions = (1 if only_pm2_5 else 18)* (dayhours - reject_first_n_hours) + 1  # another dimension for the 'b' value
    w= np.ones([dimensions,1], dtype=np.float128) #column vector
    x = np.concatenate((np.ones([total_data_entries, 1]), x), axis = 1).astype(np.float128) #create a column vector of [1,1,1,1,...], appends it to the left of the data matrix x


    adagrad = np.zeros([dimensions, 1])
    eps = 0.0000000001

    lr10 = np.empty(noitrs)

    for i in range(noitrs):
        yprime = np.dot(x,w)
        #loss = np.sum(np.power(yprime - y,2))
        loss = np.sqrt(np.sum(np.power(yprime - y, 2))/total_data_entries) 

        lr10[i] = loss
        
        gradient = 2* np.dot(x.transpose(),yprime-y)
        adagrad += gradient ** 2
        w = w - lr * gradient/np.sqrt(adagrad + eps)
        #print('finished itr %d' % i)

    print('Finish collecting data for lr = %f'% lr)

    #lr = 100
    lr = 100

    x = np.copy(x_original)

    x = x_train_set # using part of the data(training set)
    total_data_entries = np.shape(x)[0]
    y = y_train_set

    dimensions = (1 if only_pm2_5 else 18)* (dayhours - reject_first_n_hours) + 1  # another dimension for the 'b' value
    w= np.ones([dimensions,1], dtype=np.float128) #column vector
    x = np.concatenate((np.ones([total_data_entries, 1]), x), axis = 1).astype(np.float128) #create a column vector of [1,1,1,1,...], appends it to the left of the data matrix x


    adagrad = np.zeros([dimensions, 1])
    eps = 0.0000000001

    lr100 = np.empty(noitrs)

    for i in range(noitrs):
        yprime = np.dot(x,w)
        #loss = np.sum(np.power(yprime - y,2))
        loss = np.sqrt(np.sum(np.power(yprime - y, 2))/total_data_entries) 

        lr100[i] = loss
        
        gradient = 2* np.dot(x.transpose(),yprime-y)
        adagrad += gradient ** 2
        w = w - lr * gradient/np.sqrt(adagrad + eps)
        #print('finished itr %d' % i)

    print('Finish collecting data for lr = %f'% lr)

    # lr = 1000
    lr = 1000

    x = np.copy(x_original)

    x = x_train_set # using part of the data(training set)
    total_data_entries = np.shape(x)[0]
    y = y_train_set

    dimensions = (1 if only_pm2_5 else 18)* (dayhours - reject_first_n_hours) + 1  # another dimension for the 'b' value
    w= np.ones([dimensions,1], dtype=np.float128) #column vector
    x = np.concatenate((np.ones([total_data_entries, 1]), x), axis = 1).astype(np.float128) #create a column vector of [1,1,1,1,...], appends it to the left of the data matrix x


    adagrad = np.zeros([dimensions, 1])
    eps = 0.0000000001

    lr1000 = np.empty(noitrs)

    for i in range(noitrs):
        yprime = np.dot(x,w)
        #loss = np.sum(np.power(yprime - y,2))
        loss = np.sqrt(np.sum(np.power(yprime - y, 2))/total_data_entries) 

        lr1000[i] = loss
        
        gradient = 2* np.dot(x.transpose(),yprime-y)
        adagrad += gradient ** 2
        w = w - lr * gradient/np.sqrt(adagrad + eps)
        #print('finished itr %d' % i)

    print('Finish collecting data for lr = %f'% lr)

    import matplotlib.pyplot as plt

    fig,(ax1,ax2,ax3) = plt.subplots(3)
    ax1.set_xlim([1,20])
    ax1.set_ylim([0,10000])
    ax1.set_xlabel('# of iterations')
    ax1.set_ylabel('Loss function (RMS)')
    ax1.set_title('Converging process of different learning rates in linear regression (1)')
    ax1.plot(range(noitrs),lr1,label=   'lr = 1')
    ax1.plot(range(noitrs),lr10, label='lr = 10')
    ax1.plot(range(noitrs),lr100, label='lr = 100')
    ax1.plot(range(noitrs),lr1000,label='lr = 1000')
    ax1.legend()


    ax2.set_xlim([1,500])
    ax2.set_ylim([0,500])
    ax2.set_xlabel('# of iterations')
    ax2.set_ylabel('Loss function (RMS)')
    ax2.set_title('Converging process of different learning rates in linear regression (2)')
    ax2.plot(range(noitrs),lr1,label=   'lr = 1')
    ax2.plot(range(noitrs),lr10, label='lr = 10')
    ax2.plot(range(noitrs),lr100, label='lr = 100')
    ax2.plot(range(noitrs),lr1000,label='lr = 1000')
    ax2.legend()


    ax3.set_xlim([1,999])
    ax3.set_ylim([0,50])
    ax3.set_xlabel('# of iterations')
    ax3.set_ylabel('Loss function (RMS)')
    ax3.set_title('Converging process of different learning rates in linear regression (3)')
    ax3.plot(range(noitrs),lr1,label=   'lr = 1')
    ax3.plot(range(noitrs),lr10, label='lr = 10')
    ax3.plot(range(noitrs),lr100, label='lr = 100')
    ax3.plot(range(noitrs),lr1000,label='lr = 1000')
    ax3.legend()

    fig.tight_layout()
    fig.savefig(image_path)

print('End of Program')



