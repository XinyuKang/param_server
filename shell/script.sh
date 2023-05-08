for try in 1
do
	servers=1
	workers=1
	echo $try
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"



	servers=2
	workers=1
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"


	servers=4
	workers=1
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=6
	workers=1
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=8
	workers=1
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=10
	workers=1
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

done
