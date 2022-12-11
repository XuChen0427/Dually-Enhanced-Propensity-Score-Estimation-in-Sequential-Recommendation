python run_recbole.py --model=GRU4Rec --dataset=ml-1m --config_files=recbole/properties/baselines/gru4rec_ml-1m.yaml

python run_hyper.py --model=DeepFM --dataset=ml-1m --config_files=recbole/properties/baselines/DeepFM_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DeepFM.test

nohup python -u run_hyper.py --model=DeepFM --dataset=ml-1m \
--config_files=recbole/properties/baselines/DeepFM_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DeepFM.test \
>DeepFM_ml-1m_tuning.out 2>&1 &
python run_recbole.py --model=GRU4Rec2IPS --dataset=ml-1m --config_files=recbole/properties/SquenceUnBiasRec/gru4rec_ips_ml-1m.yaml
#hyper param tuning....


nohup python -u run_hyper.py --model=GRU4Rec --dataset=ml-1m --config_files=recbole/properties/baselines_full/gru4rec_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/GRU4Rec.test >GRU4Rec_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=DeepFM --dataset=ml-1m --config_files=recbole/properties/baselines_full/DeepFM_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DeepFM.test >DeepFM_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=DIN --dataset=ml-1m --config_files=recbole/properties/baselines_full/DIN_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DIN.test >DIN_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=NeuMF --dataset=ml-1m --config_files=recbole/properties/baselines_full/NCF_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/NCF.test >NCF_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=DMF --dataset=ml-1m --config_files=recbole/properties/baselines_full/DMF_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DMF.test >DMF_ml-1m_full_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=BERT4Rec --dataset=ml-1m --config_files=recbole/properties/baselines_full/bert4rec_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/bert4rec.test >bert4rec_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=Caser --dataset=ml-1m --config_files=recbole/properties/baselines_full/caser_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/caser.test >caser_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=WideDeep --dataset=ml-1m --config_files=recbole/properties/baselines_full/widedeep_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/widedeep.test >widedeep_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=TransRec --dataset=ml-1m --config_files=recbole/properties/baselines_full/transrec_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/transrec.test >transrec_full_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=GRU4Rec2IPS --dataset=ml-1m --config_files=recbole/properties/SquenceUnBiasRec/gru4rec_ips_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/GRU4Rec.test >GRU4Rec2IPS_full_ml-1m_tuning.out 2>&1 &



nohup python -u run_recbole.py --model=Twodir2BERT4Rec --dataset=ml-1m --config_files=recbole/properties/SquenceUnBiasRec/twodir2bert4rec_ml-1m.yaml >twodir2bert4rec_IPS_ml-1m.yaml 2>&1 &

python -u run_recbole.py --model=DIN --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/DIN_ml-1m.yaml

nohup python -u run_recbole.py --model=Twodir2BERT4Rec --dataset=ml-1m --config_files=recbole/properties/SquenceUnBiasRec/twodir2bert4rec_ml-1m.yaml >twodir2bert4rec_layer4_ml-1m.out 2>&1 &

nohup python -u run_hyper.py --model=GRU4Rec --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/gru4rec_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/GRU4Rec.test >GRU4Rec_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=DeepFM --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/DeepFM_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DeepFM.test >DeepFM_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=DIN --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/DIN_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DIN.test >DIN_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=NeuMF --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/NCF_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/NCF.test >NCF_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=DMF --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/DMF_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/DMF.test >DMF_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=BERT4Rec --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/bert4rec_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/bert4rec.test >bert4rec_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=Caser --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/caser_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/caser.test >caser_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=WideDeep --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/widedeep_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/widedeep.test >widedeep_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=TransRec --dataset=ml-1m --config_files=recbole/properties/baselines_labeled/transrec_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/transrec.test >transrec_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_hyper.py --model=GRU4Rec2IPS --dataset=ml-1m --config_files=recbole/properties/SquenceUnBiasRec/gru4rec_ips_ml-1m.yaml --params_file=recbole/hyper_params/ml-1m/GRU4Rec.test >GRU4Rec2IPS_labeled_ml-1m_tuning.out 2>&1 &

nohup python -u run_recbole.py --model=Twodir2BERT4Rec --dataset=mind_small --config_files=recbole/properties/SquenceUnBiasRec/twodir2bert4rec_mind.yaml >TDM_mind_0501.out 2>&1 &

nohup python -u run_recbole.py --model=Twodir2BERT4Rec --dataset=Amazon_Digital_Music --config_files=recbole/properties/SquenceUnBiasRec/twodir2bert4rec_amazon_music.yaml >TDM_music_0501.out 2>&1 &
nohup python -u run_recbole.py --model=Twodir2BERT4Rec --dataset=Amazon_Beauty --config_files=recbole/properties/SquenceUnBiasRec/twodir2bert4rec_amazon_beauty.yaml >TDM_beauty_0501.out 2>&1 &

nohup python -u run_recbole.py --model=Twodir2BERT4Rec --dataset=ml-1m --config_files=recbole/properties/SquenceUnBiasRec/twodir2bert4rec_ml-1m.yaml >TDM_ml-1m_0501.out 2>&1 &

python -u run_recbole.py --model=NeuMF --dataset=ml-1m --config_files=recbole/properties/baselines_labeled_ml-1m/NCF_ml-1m.yaml