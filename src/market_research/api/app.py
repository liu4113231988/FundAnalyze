"""FastAPI composition root. Start with uv run market-research --port 8877."""
from contextlib import asynccontextmanager
import argparse
import logging
import os
from pathlib import Path
from importlib.resources import files

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import market_research.config as C
from market_research.errors import JobsBusy
from market_research.tasks.jobs import JOBS
from market_research.tasks.processes import launch
from market_research.settings import Settings
import market_research.api.routes.analysis as analysis
import market_research.api.routes.market as market
import market_research.api.routes.meta as meta
import market_research.api.routes.sync as sync
import market_research.api.routes.tasks as tasks
from market_research.api.schemas import ErrorResponse

LOG=logging.getLogger(__name__)
STATIC=Path(str(files('webapp')/'static'))


class BodyLimit:
    """Bound JSON request bodies even when no Content-Length is supplied."""
    def __init__(self,app,limit=8*1024*1024):
        self.app,self.limit=app,limit

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method'] not in ('POST','PUT','PATCH'):
            return await self.app(scope,receive,send)
        messages=[]
        size=0
        while True:
            message=await receive()
            messages.append(message)
            size+=len(message.get('body',b''))
            if size>self.limit:
                return await JSONResponse({'ok':False,'error':'请求体超过 8 MB','field':'body'},status_code=413)(scope,receive,send)
            if message['type']=='http.disconnect' or not message.get('more_body',False):
                break
        async def replay():
            return messages.pop(0) if messages else await receive()
        await self.app(scope,replay,send)


@asynccontextmanager
async def lifespan(app):
    JOBS.migrate()
    if Settings.load().auto_services:
        # The independent singleton supervisor survives API restart. Every
        # analysis has its own process; duplicate supervisors exit on file lock.
        launch('market_research.tasks.worker')
    yield


def create_app():
    app=FastAPI(title='Market Research API',version='0.3.0',lifespan=lifespan,
                responses={400:{'model':ErrorResponse},429:{'model':ErrorResponse}})
    app.add_middleware(BodyLimit)
    origins=[v.strip() for v in os.environ.get('FIL_CORS_ORIGINS','').split(',') if v.strip()]
    if origins:
        app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])

    @app.exception_handler(RequestValidationError)
    async def validation_error(request:Request,error:RequestValidationError):
        issue=error.errors()[0]
        location=[str(x) for x in issue['loc'] if x not in ('body','query','path')]
        return JSONResponse({'ok':False,'error':issue['msg'].removeprefix('Value error, '),'field':'.'.join(location) or 'body'},status_code=400)

    @app.exception_handler(JobsBusy)
    async def busy(request:Request,error:JobsBusy):
        return JSONResponse({'ok':False,'error':str(error),'field':'concurrency'},status_code=429)

    @app.exception_handler(ValueError)
    async def invalid(request:Request,error:ValueError):
        return JSONResponse({'ok':False,'error':str(error),'field':'params'},status_code=400)

    @app.exception_handler(Exception)
    async def unexpected(request:Request,error:Exception):
        LOG.error('请求处理失败 %s',request.url.path,exc_info=error)
        return JSONResponse({'ok':False,'error':'服务处理失败，请查看运行日志后重试'},status_code=500)

    @app.get('/',include_in_schema=False)
    def index():
        return FileResponse(STATIC/'index.html')

    for module in (meta,analysis,market,sync,tasks):
        app.include_router(module.router)
    app.mount('/static',StaticFiles(directory=STATIC),name='static')
    app.mount('/vendor',StaticFiles(directory=STATIC/'vendor'),name='vendor')
    return app


app=create_app()


def main():
    settings=Settings.load()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default=settings.host)
    parser.add_argument('--port',type=int,default=settings.port)
    parser.add_argument('--debug',action='store_true')
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(app,host=args.host,port=args.port,log_level='debug' if args.debug else 'info')


if __name__=='__main__':
    main()
