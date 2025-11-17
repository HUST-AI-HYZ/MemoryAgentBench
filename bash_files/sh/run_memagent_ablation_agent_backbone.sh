source ~/.bashrc
conda activate MABench

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
root=$(pwd)


file_name=ablation_agent_backbone.txt

for line in {9..15..1}
    do
        cfg=$(sed -n "$line"p ${root}/bash_files/configs/${file_name})
        agent_config=$(echo $cfg | cut -f 1 -d ' ')
        dataset_config=$(echo $cfg | cut -f 2 -d ' ')
        backbone=$(echo $cfg | cut -f 3 -d ' ')
        max_test_queries_ablation=$(echo $cfg | cut -f 4 -d ' ')

        echo ................Start........... 
        CUDA_VISIBLE_DEVICES=7 python main.py \
                                            --agent_config      configs/agent_conf/RAG_Agents/${backbone}/${agent_config} \
                                            --dataset_config    configs/data_conf/${dataset_config} \
                                            --max_test_queries_ablation ${max_test_queries_ablation}
        echo ................End...........

    done

# bash bash_files/sh/run_memagent_ablation_agent_backbone.sh   
