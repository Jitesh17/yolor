import cv2
# pip install pyjeasy printj
import printj
from pyjeasy.check_utils import check_file_exists
from pyjeasy.file_utils import dir_contents_path_list_with_extension
from pyjeasy.image_utils import draw_bbox, show_image
from tqdm import tqdm

label_map = {"0": "H"}  # elmet


class Annotation:
    def __init__(self, label, xmin, ymin, width, height) -> None:
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.height = height
        self.width = width
        self.xmax = xmin + width
        self.ymax = ymin+height

    @classmethod
    def from_str(cls, str):
        label, xmin, ymin, width, height = str.replace("\n", "").split(' ')
        return cls(label, xmin, ymin, width, height)

    @classmethod
    def from_text_str(cls, str, shape):
        label, xmin, ymin, width, height = str.replace("\n", "").split(' ')
        xmin, ymin, width, height = float(xmin), float(
            ymin), float(width), float(height)
        xmin = xmin - width/2
        ymin = ymin - height/2
        return cls(label, int(xmin*shape[1]), int(ymin*shape[0]), int(width*shape[1]), int(height*shape[0]))

    def __str__(self):
        return f"label: {self.label}, xmin: {self.xmin}, ymin: {self.ymin}, xmax: {self.xmax}, ymax: {self.ymax}, width: {self.width}, height: {self.height}"

    def __repr__(self) -> str:
        return self.__str__()


def preview_annotations(img_path_list, show_image_preview: bool = True, write_image: bool = False):
    for img_path in tqdm(img_path_list):
        ann_path = f"{root_path}/labels/{img_path.split('/')[-1].split('.')[0]}.txt"
        check_file_exists(ann_path)
        img = cv2.imread(img_path)
        ann_list = []
        printj.cyan(f"img: {img_path}")
        printj.cyan(f"txt: {ann_path}")
        with open(ann_path) as f:
            lines = f.readlines()
            for line in lines:
                ann = Annotation.from_text_str(line, img.shape)
                printj.yellow(ann)
                ann_list.append(ann)
                img = draw_bbox(
                    img, [ann.xmin, ann.ymin, ann.xmax, ann.ymax], text=label_map[ann.label])
            if show_image_preview:
                show_image(img)
            if write_image:
                cv2.imwrite(
                    f"{root_path}/preview_annotations/{img_path.split('/')[-1]}", img)


def split_data(img_path_list, ratio: list):
    """Split dataset

    Args:
        img_path_list ([type]): list of img paths of all datasets
        ratio (list): train: val: test

    Returns:
        tuple[list[str]]: [description]
    """
    from sklearn.model_selection import train_test_split
    _sum = sum(ratio)
    test_ratio = ratio[-1]/_sum
    train, test = train_test_split(img_path_list, test_size=test_ratio)
    if len(ratio) == 3:
        train, val = train_test_split(
            train, test_size=ratio[1]/_sum/(1-test_ratio))
        return train, val, test
    return train, test


def write_data(img_path_list, ratio: list):

    data = split_data(img_path_list, ratio)
    printj.cyan(
        f"all: {len(img_path_list)}, train: {len(data[0])}, val: {len(data[1])}, test: {len(data[2])}")

    for i, name in enumerate(["train", "val", "test"]):
        f = open(f"helmet_dataset/{name}.txt", "w")
        for line in data[i]:
            f.write(line+"\r")
        f.close()


if __name__ == '__main__':

    root_path = "helmet_dataset"

    img_path_list = dir_contents_path_list_with_extension(
        f"{root_path}/images", [".png"])
    # preview_annotations(img_path_list)
    write_data(img_path_list, [90, 5, 5])
