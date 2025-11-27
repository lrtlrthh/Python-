import os
from tqdm import tqdm
import cv2


def main(path, goal_path):
    lis = os.listdir(path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    save_images = []
    save_images_path = []
    for sex in lis:
        img_list = os.listdir(path + '/' + sex)
        idx = 0
        for img in tqdm(img_list):
            image = cv2.imread(path + '/' + sex + '/' + img)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # 绘制人脸矩形框
            save_img = []
            save_path = []
            for (x, y, w, h) in faces:
                # image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
                save_image = image[y + 1:y + h, x + 1:x + w]
                save_img.append(save_image)
                save_path.append(goal_path + '/' + 'train_data' + '/' + sex + '/' + f'{idx}.jpg')
                idx += 1
            # cv2.imshow('show', image)

            save_images += save_img
            save_images_path += save_path

    save(save_images, save_images_path)


def save(data, path):
    count = len(data)
    print(count)
    for i in tqdm(range(count)):
        cv2.imwrite(path[i], data[i])


if __name__ == '__main__':
    data_source = 'ori_data'
    goal_path = 'data'
    main(data_source, goal_path)
