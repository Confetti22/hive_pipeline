teacher_dpt:

python read_metrics_from_json.py /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/dpt_linear_probTrue_batch16_teacher_dpt/valid_metrics.jsonl  /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/s_tinyvit_linear_probTrue_batch16_distill_aug_rm009_e500/valid_metrics.jsonl --names 'teacher_DINOV3' 'student_tinyvit'   --epoch 45 --prefix valid


python read_metrics_from_json.py \
  /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/dpt_linear_probTrue_batch16_teacher_dpt/valid_metrics.jsonl \
  /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/s_tinyvit_linear_probTrue_batch16_t1779_no_norm_wd_e100/valid_metrics.jsonl\
  /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/cmpsd_scratch_batch16_96_48/valid_metrics.jsonl \
  /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/cmpsdlinear_prob_contrastive_pretrained_batch16_one_stage_cosloss_rm009_4um_e100/valid_metrics.jsonl\
  /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/s_tinyvit_scratch_batch16/valid_metrics.jsonl \
  --names teacher_linear_prob student_linear_prob cnn_scratch contrastive_pretrained student_scratch \
  --epoch 45 \
  --prefix valid
  
  python read_metrics_from_json.py \
    /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/cmpsd_scratch_batch16_96_48/valid_metrics.jsonl \
    /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/s_tinyvit_scratch_batch16/valid_metrics.jsonl \
  --names 'cnn_scratch' 'student_scratch' \
  --epoch 150 \
  --prefix valid
  
 
  s_tinyvit_linear_probTrue_batch16_t1779_no_norm_wd_e100
 
  s_tinyvit_linear_probTrue_batch16_rm009_e100

    python read_metrics_from_json.py \
    /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/s_tinyvit_linear_probTrue_batch16_t1779_no_norm_wd_e100/valid_metrics.jsonl \
    /home/confetti/e5_workspace/hive1_pipeline/runs/seg_dino/s_tinyvit_linear_probTrue_batch16_rm009_e100/valid_metrics.jsonl \
  --names 't1779_distilled' 'rm009_distilled' \
  --epoch 45 \
  --prefix valid
  

  