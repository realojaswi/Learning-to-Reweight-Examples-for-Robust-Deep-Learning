module load use.own
module load conda-env/mypackages-py3.8.5

#for ((i = 0 ; i < 101 ; i=i+10 )); 
#do 
#    python main.py --noise_ratio "${i}" &
#    sleep 1;
#done


for ((i = 0 ; i < 101 ; i=i+10 )); 
do 
    python main.py --class_ratio "${i}" --noise_ratio 10 &
    sleep 1;
done

