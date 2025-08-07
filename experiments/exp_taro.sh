thread_count=("1" "2" "4")

mkdir data

rm -rf data/taro
mkdir data/taro

#---------------------
# fully disconnected
#---------------------

rm -rf data/taro/fd
mkdir data/taro/fd

# use num_itr = 10
for tc in ${thread_count[@]}
do
  cd ../build

  for matrix_size in 50 100 150 
  do
    for num_nodes in 1000 2000 3000 4000 5000 
    do
      echo "writing results for fully disconnected, thread count = ${tc}, matrix_size = ${matrix_size}, num_nodes = ${num_nodes}"
      for i in 1 2 3 
      do
        ./examples/fd_tf 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/fd/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_tf.txt
        ./examples/fd_central_queue 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/fd/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_central_queue.txt
        ./examples/fd_central_priority_queue 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/fd/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_priority_queue.txt
        ./examples/fd_taro 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/fd/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_taro.txt
      done
      echo "done!"
    done
  done
  cd ../experiments
done


#---------------------
# linear chain 
#---------------------

rm -rf data/taro/lc
mkdir data/taro/lc

# use num_itr = 10
for tc in ${thread_count[@]}
do
  cd ../build

  for matrix_size in 50 100 150 
  do
    for num_nodes in 1000 2000 3000 4000 5000 
    do
      echo "writing results for linear chain, thread count = ${tc}, matrix_size = ${matrix_size}, num_nodes = ${num_nodes}"
      for i in 1 2 3 
      do

        ./examples/lc_tf 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/lc/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_tf.txt
        ./examples/lc_central_queue 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/lc/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_central_queue.txt
        ./examples/lc_central_priority_queue 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/lc/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_priority_queue.txt
        ./examples/lc_taro 20 ${matrix_size} ${num_nodes} ${tc} >> ../experiments/data/taro/lc/tc_${tc}_ms_${matrix_size}_nn_${num_nodes}_taro.txt
      done
      echo "done!"
    done
  done
  cd ../experiments
done
