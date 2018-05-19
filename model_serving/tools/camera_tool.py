# !/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (c) luoadore, Inc. All Rights Reserved
#
"""
This module provide
    Authors: xiaobing
Date:	2018/5/19
"""

import cv2
from threading import Thread
from multiprocessing import Queue, Pool


class WebcamVideoStream(object):
    """
    封装open_cv的视频帧读取，提高性能
    """

    def __init__(self, width, height):
        """
        初始化摄像头，并从视频流中读取一帧图像
        :param width: 设置读取的视频帧的宽度
        :param height: 设置读取视频帧的高度
        """
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        # 视频读取线程默认不终止
        self.stopped = False

    def start(self):
        """
        开始读取视频帧
        :return:
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        线程终止前不断从流中读取视频并更新
        :return:
        """
        while True:
            # 当self.stoped参数设置为True时，终止采集
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """
        返回最新的一帧
        :return:
        """
        return self.frame

    def stop(self):
        """
        将停止标志位置True，停止读视频流
        :return:
        """
        self.stopped = True

    # @staticmethod
    # def canmra_detect_ontime():
    #         # 检测逻辑
    #         # 多进程输入队列
    #         input_q = Queue(1000)
    #         # 多进程输出队列
    #         output_q = Queue(1000)
    #         # 多进程加载模型
    #         pool = Pool(5, worker, (input_q, output_q))
    #         video_capture = WebcamVideoStream(640, 426).start()
    #
    #         while True:
    #             # video_capture多线程读取视频流
    #             frame = video_capture.read()
    #             # 视频帧放入多进程输入队列
    #             input_q.put(frame)
    #             # 多进程输出队列取出标记好物体的图片
    #             frame = output_q.get()
    #             # 展示已标记物体的图片
    #             cv2.imshow('Video', frame)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #
    #         # 关闭多进程
    #         pool.terminate()
    #         # 关闭视频流
    #         video_capture.stop()
    #         # opencv窗口关闭
    #         cv2.destroyAllWindows()

    @staticmethod
    def test_camera():
                # capture from camera at location 0
                cap = cv2.VideoCapture(0)
                # set the width and height, and UNSUCCESSFULLY set the exposure time
                cap.set(3, 640)
                cap.set(4, 426)
                # cap.set(15, 0.1)

                while True:
                    ret, img = cap.read()
                    cv2.imshow("input", img)
                    # cv2.imshow("thresholded", imgray*thresh2)

                    key = cv2.waitKey(10)
                    if key == 27:
                        break

                cv2.destroyAllWindows()
                cv2.VideoCapture(0).release()
