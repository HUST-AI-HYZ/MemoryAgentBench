source ~/.bashrc
source /scratch/yzhu/anaconda3/etc/profile.d/conda.sh
conda activate MABench

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
root=$(pwd)


file_name=long_context_agents.txt

for line in {5..19..1}
    do 
        cfg=$(sed -n "$line"p ${root}/bash_files/configs/${file_name})
        agent_config=$(echo $cfg | cut -f 1 -d ' ')
        dataset_config=$(echo $cfg | cut -f 2 -d ' ')

        echo ................Start........... 
        CUDA_VISIBLE_DEVICES=7 python main.py \
                                    --agent_config      configs/agent_conf/Long_Context_Agents/${agent_config} \
                                    --dataset_config    configs/data_conf/${dataset_config} 
        echo ................End...........

    done

# bash bash_files/sh/run_memagent_longcontext.sh   
