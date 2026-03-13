'''应用启动入口：负责组装配置、服务实例、路由和中间件。'''

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import annotate, batch, infer, jobs, review
from app.core.config import get_settings
from app.core.database import init_db
from app.services.batch_service import BatchService
from app.services.inference_service import InferenceService
from app.services.job_service import InMemoryJobStore
from app.services.review_service import ReviewService


 # 创建并装配 FastAPI 应用实例
def create_app() -> FastAPI:
    settings = get_settings()

    # 生命周期钩子：在应用启动时注入共享服务，在关闭时统一释放上下文
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()  # 启动时自动建表，确保阶段C数据闭环可用。

        inference_service = InferenceService(settings)  # 推理服务（懒加载模型）
        job_store = InMemoryJobStore()  # 任务状态存储（内存实现）
        review_service = ReviewService(settings)
        batch_service = BatchService(
            inference_service=inference_service,
            job_store=job_store,
            review_service=review_service,
        )

        # 将核心对象挂载到 app.state，供各路由复用
        app.state.settings = settings
        app.state.inference_service = inference_service
        app.state.job_store = job_store
        app.state.batch_service = batch_service
        app.state.review_service = review_service

        # 生产可改为 QA_LAZY_LOAD_MODELS=false，在启动阶段提前加载模型，做到失败早发现。
        if not settings.lazy_load_models:
            inference_service.initialize()
        yield

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    # 配置跨域策略，支持前端页面跨域访问 API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=settings.allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 健康检查接口：用于探活与运行状态观察
    @app.get("/api/health")
    def health_check():
        return {
            "status": "ok",
            "lazy_load_models": settings.lazy_load_models,
            "model_initialized": app.state.inference_service.is_initialized,
            "job_store_size": app.state.job_store.total_jobs(),
        }

    # 注册业务路由
    app.include_router(infer.router)
    app.include_router(batch.router)
    app.include_router(jobs.router)
    app.include_router(review.router)
    app.include_router(annotate.router)
    return app


# 导出给 Uvicorn 使用的应用对象
app = create_app()
