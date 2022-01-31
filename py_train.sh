python train.py part_A_train.json part_A_test.json 1 3 | tee train_3.log
# python train.py part_A_train.json A_test_data.json 2 0 | tee train_0.log
# python train.py part_A_train_ShanghaiTech.json part_A_test_ShanghaiTech.json 2 1 | tee train_0.log
# python train.py part_A_train_ShanghaiTech.json part_A_test_ShanghaiTech.json -p 2model_best.pth.tar 0 2 | tee train_2.log