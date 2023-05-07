for try in 1 2 3
do
	servers=1
	workers=1
	echo $try
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"



	servers=1
	workers=2
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"


	servers=1
	workers=4
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=1
	workers=4
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=1
	workers=8
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=1
	workers=10
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=4
	workers=2
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=2
	workers=4
	python main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

done

bash script_workers_serv_inc.sh

