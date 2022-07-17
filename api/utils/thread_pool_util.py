# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: thread_pool_utils.py
@Time: 2022/7/17 1:05
"""
# 线程池
# 解释：
# 1）一个工人同一时间只做一个任务，但做完一个任务可以接着做下一个任务；
# 2）可以分配多个任务给少量工人，减少人员成本开销。
# 3）任务按顺序分配给空闲工人，但每个任务的耗时不一样，任务不是按顺序被完成的，后提交的任务可能会先被完成

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor


# 任务
def task(taskId, consuming):
    thread_name = threading.current_thread().getName()
    print('工人【%s】正在处理任务【%d】：do something...' % (thread_name, taskId))
    # 模拟任务耗时(秒)
    time.sleep(consuming)
    print('任务【%d】：done' % taskId)


def main():
    # 5个工人
    pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix='Thread')
    # 准备10个任务
    for i in range(10):
        # 模拟任务耗时(秒)
        consuming = random.randint(1, 5)
        pool.submit(task, i+1, consuming)


if __name__ == '__main__':
    main()