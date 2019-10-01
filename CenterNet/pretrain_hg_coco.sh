cd ../src

load_model='/usr/zll/person_reid/code/CenterNet/models/ExtremeNet_500000.pth'
train_annot_path='/usr/zll/person_reid/code/CenterNet/data/kc/level2/annotations/level1_drone_day_no_small_obj_4373.json'
test_annot_path='/usr/zll/person_reid/code/CenterNet/data/kc/level2/annotations/level2_test_b1-b5_day_part.json'

python main.py ctdet --exp_id pretrain_hg_coco --dataset coco --arch hourglass --batch_size 8 --lr 2.5e-5 \
  --gpus 0,1,2,3 --num_epochs 20 --train_annot_path $train_annot_path --load_model $load_model --test_annot_path $test_annot_path

cd ..