accelerate launch --num_processes=2 scripts/evaluate.py --model_family llava-v15 --model_id llava-v1.5-7b --dataset.type vizwiz-slim --dataset.root_dir ./data
python scripts/evaluate.py --model_family llava-v15 --model_id llava-v1.5-7b --dataset.type vizwiz-slim --dataset.root_dir ./data

accelerate launch --num_processes=2 scripts/evaluate.py --model_id prism-dinosiglip+7b --dataset.type text-vqa-slim --dataset.root_dir ./data
python scripts/evaluate.py --model_id prism-dinosiglip+7b --dataset.type text-vqa-slim --dataset.root_dir ./data

for dataset_id in 'vqa-v2-slim' 'gqa-slim' 'vizwiz-slim' 'text-vqa-slim' 'nocaps-slim' 'pope-slim' 'refcoco-slim' 'ocid-ref-slim' 'tally-qa-slim'; do CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --model_id prism-dinosiglip+7b --dataset.type $dataset_id --dataset.root_dir ./data; done
for dataset_id in 'vqa-v2-slim' 'gqa-slim' 'vizwiz-slim' 'text-vqa-slim' 'nocaps-slim' 'pope-slim' 'refcoco-slim' 'ocid-ref-slim' 'tally-qa-slim'; do CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --model_family llava-v15 --model_id llava-v1.5-7b --dataset.type $dataset_id --dataset.root_dir ./data; done


