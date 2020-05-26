echo "Checking the Accuracy\n"
value=$(<AccuracyCNN.txt)
echo "Accuracy is : $value\n"

if [ "$value" != 80 ]

then 
	sed -i 's/filters2=.*/filters2=32/' /pythoncnn/cnn2.py
	sed -i 's/units1=.*/units1=256/' /pythoncnn/cnn2.py
	sed -i 's/steps_per_epoch1=.*/steps_per_epoch1=1284/' /pythoncnn/cnn2.py
	sed -i 's/validation_steps1=.*/validation_steps1=4297/' /pythoncnn/cnn2.py
	sed -i 's/epochs1=.*/epochs1=3/' /pythoncnn/cnn2.py
	python3 /pythoncnn/tweaking.py
	echo "Hyper Parameter Changed successful"

else
       	echo "accuracy is $value so no need of changing the hyper parameter"
fi

