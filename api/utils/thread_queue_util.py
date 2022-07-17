# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: thread_queue_util.py
@Time: 2022/7/17 1:13
"""
# 线程队列
# 解释：
# 1）一个队列有N个工人在排队，按队列排序给他们分配任务；
# 2）做得再快，也要按排队排序来接任务，不能插队抢任务。

# @see https://docs.python.org/zh-cn/3/library/queue.html#queue-objects

import time
import random
import threading
from queue import Queue


# 自定义线程
class CustomThread(threading.Thread):
    def __init__(self, queue, **kwargs):
        super(CustomThread, self).__init__(**kwargs)
        self.__queue = queue

    def run(self):
        while True:
            # (工人)获取任务
            item = self.__queue.get()
            # 执行任务
            item[0](*item[1:])
            # 告诉队列，任务已完成
            self.__queue.task_done()


# 任务
def task(taskId, consuming):
    thread_name = threading.current_thread().getName()
    print('工人【%s】正在处理任务【%d】：do something...' % (thread_name, taskId))
    # 模拟任务耗时(秒)
    time.sleep(consuming)
    print('任务【%d】：done' % taskId)


def main():
    q = Queue()
    # 招工，这里招了5个工人(启动5个线程)
    for i in range(5):
        t = CustomThread(q, daemon=True)
        # 工人已经准备好接活了
        t.start()

    # 来活了(往队列里塞任务)
    for i in range(10):
        taskId = i + 1
        # 模拟任务耗时(秒)
        consuming = random.randint(1, 5)
        q.put((task, taskId, consuming))

    # 阻塞队列
    q.join()


if __name__ == '__main__':
    main()