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



	servers=1
	workers=2
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"


	servers=1
	workers=4
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=1
	workers=4
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=1
	workers=8
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=1
	workers=10
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=4
	workers=2
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

	servers=2
	workers=4
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 400 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers".json"

done


