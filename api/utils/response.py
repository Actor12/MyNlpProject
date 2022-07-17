# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: response.py
@Time: 2022/7/16 18:52
"""
import warnings

warnings.filterwarnings("ignore")
from fastapi import status
from fastapi.responses import JSONResponse, Response
from typing import Union


def resp_200(*, data: Union[list, dict, str]) -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': 200,
            'message': "Success",
            'data': data,
        }
    )


def resp_400(*, data: str = None, message: str = "BAD REQUEST") -> Response:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            'code': 400,
            'message': message,
            'data': data,
        }
    )


def response(*, code=200, data: Union[list, dict, str], message="Success") -> Response:
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            'code': code,
            'message': message,
            'data': data,
        }
    )
