k=$1
n=$2
b=$3
echo "SBSUR:"
{ command time --verbose python experiments/run_sbsur.py $k $n $b ; } 2> time.log
cat time.log | grep -e "wall clock" -e "Maximum"
echo "UR:"
{ command time --verbose python experiments/run_ur.py $k $n $b ; } 2> time.log
cat time.log | grep -e "wall clock" -e "Maximum"
rm time.log