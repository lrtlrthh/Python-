import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn.functional as f
from PIL import Image
from torchvision import transforms

plt.rcParams['font.family'] = ['SimHei']


def detection_one(test_image, model_path, classifier='haarcascade_frontalface_default.xml', transform=None):
    face_cascade = cv2.CascadeClassifier(classifier)
    if test_image[-3:] == 'jpg':
        ori_image = cv2.imread(test_image)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(ori_image, 1.3, 5)
        if len(faces) == 0:
            print('未检测到人脸')
            return
        elif len(faces) == 1:
            # 画出检测结果
            detection_image = None
            for (x, y, w, h) in faces:
                ori_image = cv2.rectangle(ori_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                detection_image = ori_image[y + 2:y + h, x + 2:x + w]

            # 选出检测区域进行识别
            show_image = ori_image
            detection_image = Image.fromarray(detection_image)

            model = torch.load(model_path)
            image = detection_image

            if transform is None:
                transform = transforms.Compose([
                    transforms.Resize(224, ),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5), ])
            else:
                transform = transform

            classes = ['女', '男']
            image = transform(image)
            image = image.unsqueeze(0)
            y_ = model(image)
            pred = torch.max(y_, dim=1)[1]
            # print(image.shape)
            print(classes[int(pred)])

            plt.figure()
            plt.title(f'预测结果:{classes[int(pred)]}')
            plt.imshow(show_image)
            plt.axis('off')
            plt.show()
        else:
            print('检测到多个人脸')
            return
    elif test_image[:6] == 'camera':
        capture = cv2.VideoCapture(int(test_image[-1]))
        model = torch.load(model_path)
        model.eval()
        while True:
            c = cv2.waitKey(1)
            if c == 27:
                break
            res, frame = capture.read()
            ori_image = frame
            show_image = ori_image
            faces = face_cascade.detectMultiScale(ori_image, 1.3, 5)

            if len(faces) == 0:
                print('未检测到人脸')
            elif len(faces) == 1:
                # 画出检测结果
                detection_image = None
                for (x, y, w, h) in faces:
                    ori_image = cv2.rectangle(ori_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    detection_image = ori_image[y + 2:y + h, x + 2:x + w]

                # 选出检测区域进行识别
                show_image = ori_image
                detection_image = Image.fromarray(detection_image)
                image = detection_image

                if transform is None:
                    transform = transforms.Compose([
                        transforms.Resize(224, ),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5), ])
                else:
                    transform = transform

                classes = ['woman', 'man']
                image = transform(image)
                image = image.unsqueeze(0)
                y_ = model(image)
                Soft = f.softmax(y_).detach().numpy()[0]
                pred = torch.max(y_, dim=1)[1]

                # print(image.shape)
                font = cv2.FONT_HERSHEY_SIMPLEX
                show_image = cv2.putText(show_image, classes[int(pred)], (100, 100), font, 2, (0, 255, 0), 3)
                print('置信度', Soft[int(pred)])
                print(classes[int(pred)])
            else:
                print('检测到多个人脸')
            cv2.imshow('show_result', show_image)
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detection_one('kun.jpg', 'models/best_resnet18.pt')
    # detection_one('camera0', 'models/last_resnet34.pt')
