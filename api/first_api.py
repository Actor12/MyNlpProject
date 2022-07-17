# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: first_api.py
@Time: 2022/7/16 18:17
"""
import os
import warnings
from fastapi import FastAPI

from api.utils.response import response, resp_200, resp_400
from entity.first_api_entity import TextEntity

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
app = FastAPI()

@app.post("/first_api")
async def semantic_match_on_text_pair(request: TextEntity):
    try:
        '''在此位置封装功能'''
        res = []

    except Exception as err:
        return resp_400(message=str(err))
    else:
        return response(data=res)


if __name__ == '__main__':
    #file = os.path.split(__file__)[-1].split(".")[0]
    file = "first_api"
    os.system("uvicorn " + file + ":app --host '0.0.0.0' --port 18081 --reload")
