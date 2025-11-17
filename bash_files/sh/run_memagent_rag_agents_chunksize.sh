source ~/.bashrc
conda activate MABench

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
root=$(pwd)


file_name=rag_agents_chunksize.txt

for line in {5..5..1}
    do 
        cfg=$(sed -n "$line"p ${root}/bash_files/configs/${file_name})
        agent_config=$(echo $cfg | cut -f 1 -d ' ')
        dataset_config=$(echo $cfg | cut -f 2 -d ' ')
        chunk_size_ablation=$(echo $cfg | cut -f 3 -d ' ')


        echo ................Start........... 
        CUDA_VISIBLE_DEVICES=7 python main.py \
                                    --agent_config        configs/agent_conf/RAG_Agents/gpt-4o-mini/${agent_config} \
                                    --dataset_config      configs/data_conf/${dataset_config} \
                                    --chunk_size_ablation ${chunk_size_ablation}
        echo ................End...........

    done

# bash bash_files/sh/run_memagent_rag_agents_chunksize.sh   
