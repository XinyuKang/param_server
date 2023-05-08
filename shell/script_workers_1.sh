for try in 1
do
	servers=1
	workers=6
	echo $try
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"




done

