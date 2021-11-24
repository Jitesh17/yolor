python test.py --data data/coco_helmet.yaml \
--img 1280 --batch 1 \
--conf 0.25 --iou 0.65 --device 0 \
--cfg cfg/yolor_p6.cfg --weights runs/train/yolor_p6_1280_h1/weights/best.pt \
--name yolor_p6_val1 --names data/coco_helmet.names
