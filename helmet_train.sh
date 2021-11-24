python train.py --batch-size 1 \
--img 1280 1280 --data data/coco_helmet.yaml --cfg cfg/yolor_p6_helmet.cfg \
--weights 'runs/train/yolor_p6_1280_h1/weights/best.pt' --device 0 \
--name yolor_p6_1280_h2 --hyp data/hyp.helmet.1280.yaml \
--epochs 30

# yolor_p6.pt