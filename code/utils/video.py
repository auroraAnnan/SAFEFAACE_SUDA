from threading import Thread, Lock
from datetime import datetime
import time
import cv2

time_cycle = 80
# import the necessary packages
import sys
from queue import Queue


class FileVideoStream:
    def __init__(self, path, video_save_path="", queueSize=128, video_fps=25):
        # 初始化文件视频流以及用于指示线程是否应该停止的布尔值
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        # 初始化存储视频文件帧的队列
        self.Q = Queue(maxsize=queueSize)
        self.video_save_path = video_save_path
        self.video_fps = video_fps

    def start(self):
        # 启动一个线程从文件视频流中读取帧
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # 循环
        while True:
            # 如果设置了线程指示器变量，则停止线程
            if self.stopped:
                return
            # 否则，请确保队列中有空间
            if not self.Q.full():
                # 从文件中读取下一帧
                (grabbed, frame) = self.stream.read()
                # 如果 grabbed 布尔值为 False，那么我们已经到了视频文件的末尾
                if not grabbed:
                    self.stop()
                    return
                # 将帧添加到队列中
                self.Q.put(frame)

    def read(self):
        # 返回队列中的下一帧
        return self.Q.get()

    def more(self):
        # 如果队列中还有帧，则返回 True
        return self.Q.qsize() > 0

    def stop(self):
        # 指示应该停止线程
        self.stopped = True

    def save(self):
        # 保存视屏文件
        if self.video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(self.video_save_path, fourcc, self.video_fps, size)
            return out


class CameraThread(Thread):
    def __init__(self, src=0, video_save_path="", video_fps=25, width=640, height=480):
        # 初始化摄像机流并从流中读取第一帧
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        # 初始化用于指示线程是否应该停止的变量
        self.stopped = False
        self.video_save_path = video_save_path
        self.video_fps = video_fps


    def start(self):
        # 启动线程从视频流中读取帧
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # 继续无限循环，直到线程停止
        while True:
            # 如果设置了线程指示器变量，则停止线程
            if self.stopped:
                return
            if not self.grabbed:
                return
            # 否则，从流中读取下一帧
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # 返回最近读取的帧
        return self.grabbed, self.frame

    def stop(self):
        # 表示应该停止线程
        self.stopped = True

    def save(self):
        # 保存视屏文件
        if self.video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(self.video_save_path, fourcc, self.video_fps, size)
            return out

    def release(self):
        self.stream.release()


