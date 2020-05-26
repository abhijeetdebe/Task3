line_number1 = 34

line_number2 = 46

with open('/pythoncnn/cnn2.py','r+') as fh :
    lines = fh.readlines()
    fh.seek(0)
    lines.insert(line_number1 - 1,'model.add(Convolution2D(filters=64,kernel_size=(3,3),activation=activationconvo))\n')
    fh.writelines(lines)
    fh.seek(0)
    lines.insert(line_number2 - 1,'model.add(Dense(units=128,activation=actiavtiondense))\n')
    fh.writelines(lines)


