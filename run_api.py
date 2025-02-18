import uvicorn
import logging

if __name__ == "__main__":
    # 配置uvicorn的日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="debug",
        reload=True  # 开发模式下启用热重载
    )