import time
from threading import Event
import cv2
from retinaface import Retinaface
from utils.video import CameraThread, FileVideoStream

retinaface = Retinaface()
# --------------------------------------------------------------------------------#
#   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
#   想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
#   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
#   想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
#   video_fps用于保存的视频的fps
#   video_path、video_save_path和video_fps仅在mode='video'时有效
#   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
# ----------------------------------------------------------------------------------------------------------#
video_path = 0
video_save_path = ""
video_fps = 25.0
# -------------------------------------------------------------------------#
#   test_interval用于指定测量fps的时候，图片检测的次数
#   理论上test_interval越大，fps越准确。
# -------------------------------------------------------------------------#
test_interval = 100
# -------------------------------------------------------------------------#
#   dir_origin_path指定了用于检测的图片的文件夹路径
#   dir_save_path指定了检测完图片的保存路径
#   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
# -------------------------------------------------------------------------#
dir_origin_path = "img/"
dir_save_path = "img_out/"

# ----------------------------------------------------------------------------------------------------------#
#   mode用于指定测试的模式：
#   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
#   'video'表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
#   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
#   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
#   'register'表示注册身份信息
# ----------------------------------------------------------------------------------------------------------#

while True:
    mode = input("请选择(1.注册/2.验证/3.退出):")
    if mode == '1':
        mode = 'register'
    elif mode == '2':
        mode = 'video'
    elif mode == '3':
        print("程序退出")
        break

    if mode == 'register':
        if video_path == 0:
            kill_event = Event()
            capture = CameraThread(video_save_path=video_save_path,
                                   video_fps=video_fps)
            capture.start()
        else:
            capture = FileVideoStream(video_path=video_path, video_save_path=video_save_path,
                                      video_fps=video_fps)
            capture.start()
        # 提高fps 结束
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        (width, length, depth) = frame.shape

        fps = 0.0
        real = 0
        flat = 0
        while True:
            # image_cropper = CropImage()
            t1 = time.time()
            # 读取某一帧;
            ref, frame = capture.read()
            frame = frame[int(width * 0.2):int(width * 0.8), int(length * 0.3):int(length * 0.7), :]
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 进行检测
            # frame = np.array(retinaface.detect_image(frame))
            b, crop_img, Flag, image_box, frame = retinaface.getFace(frame)

            if Flag:
                label, score = retinaface.isReal(frame, image_box)
                if label:
                    real += 1
                    if real >= 20:
                        print("人脸识别成功,请按回车键确认注册!")
                        c = cv2.waitKey() & 0xff
                        if c == 13:
                            cv2.namedWindow("register", cv2.WINDOW_KEEPRATIO)
                            cv2.imshow("register", frame)
                            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                            name = input("用户名: ")
                            retinaface.register(crop_img, name)
                            capture.release()
                            break
                else:
                    real = 0

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.namedWindow("register", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("register", frame)
            c = cv2.waitKey(1) & 0xff

            if c == 27:
                capture.release()
                break
        if not flat:
            flat = 0
            print("Register Done!")
        capture.release()
        cv2.destroyAllWindows()

    elif mode == "video":
        # 提高fps 开始
        if video_path == 0:
            capture = CameraThread(video_save_path=video_save_path,
                                   video_fps=video_fps)
            capture.start()
        else:
            capture = FileVideoStream(video_path=video_path, video_save_path=video_save_path,
                                      video_fps=video_fps)
            capture.start()
        if video_save_path != "":
            out = capture.save()
        # 提高fps 结束
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        (width, length, depth) = frame.shape

        fps = 0.0
        real = 0
        while True:
            t1 = time.time()
            # 读取某一帧;
            ref, frame = capture.read()
            frame = frame[int(width * 0.2):int(width * 0.8), int(length * 0.3):int(length * 0.7), :]
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 进行检测
            b, crop_img, Flag, image_box, frame = retinaface.getFace(frame)

            if Flag:
                label, score = retinaface.isReal(frame, image_box)
                if label == 1:
                    real += 1
                    if real == 20:
                        frame, name = retinaface.identify(b, crop_img, frame)
                        real = 0
                        if name == 'Unknown':
                            print('验证失败,身份未注册')
                        else:
                            print('验证成功')
                        capture.release()
                        break
                else:
                    print('警告！非活体')
                    real = 0
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            cv2.namedWindow("video", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify the correct mode: '1.注册', '2.验证', '3.退出'.")
