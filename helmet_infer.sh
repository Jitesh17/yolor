# python detect.py --source ../helmet_dataset/images/1.png \
python detect.py --source ../test_video.mp4 \
--cfg cfg/yolor_p6_helmet.cfg \
--weights runs/train/yolor_p6_1280_h1/weights/best.pt --conf 0.25 \
--img-size 1280 --device 0 \
--output inference/helmet25 \
--names data/coco_helmet.names