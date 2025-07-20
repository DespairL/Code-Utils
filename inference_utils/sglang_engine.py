# SGLang Engine Utils - 可复用的推理引擎工具类
import asyncio
import sglang as sgl
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Iterator
import logging
import os
import torch

if is_in_ci():
    import patch
else:
    import nest_asyncio
    nest_asyncio.apply()

class SGLangEngine:
    """
    可复用的SGLang推理引擎封装类
    
    支持的功能:
    - 灵活的GPU设备配置 (支持指定GPU ID列表、张量并行、数据并行)
    - 多种推理模式 (同步/异步、流式/非流式、单个/批量)
    - 内存管理和优化参数 (内存分配、批处理大小、缓存配置)
    - 模型优化选项 (量化、torch.compile、CUDA图、注意力后端)
    - 统一推理接口 (支持所有推理模式的统一调用)
    - 动态配置更新 (运行时GPU配置切换)
    
    推理模式支持:
    1. 同步非流式: generate(), generate_text_only()
    2. 同步流式: generate_stream(), generate_stream_iter()
    3. 异步非流式: async_generate(), async_generate_text_only()
    4. 异步流式: async_generate_stream(), async_generate_stream_raw()
    5. 批量同步: batch_generate(), batch_generate_text_only()
    6. 批量异步: async_batch_generate(), async_batch_generate_text_only()
    7. 统一接口: infer() - 支持所有模式的统一调用
    
    GPU配置方式:
    1. 直接指定: gpu_ids=[0, 1, 2]
    2. 范围指定: base_gpu_id=1, tp_size=2, gpu_id_step=1 -> [1, 2]
    3. 动态更新: update_gpu_config([2, 3, 4])
    
    Example:
        # 基本使用
        with SGLangEngine(model_path="/path/to/model", gpu_ids=[0, 1]) as engine:
            result = engine.generate_text_only("Hello, world!")
        
        # 异步流式生成
        async for chunk in engine.async_generate_stream("Tell me a story"):
            print(chunk, end="")
        
        # 批量生成
        results = engine.batch_generate_text_only(["prompt1", "prompt2"])
        
        # 统一接口
        result = engine.infer("prompt", stream=True, async_mode=True)
    """
    
    def __init__(
        self,
        model_path: str,
        # 设备和并行配置
        device: str = "cuda",
        tp_size: int = 1,
        dp_size: int = 1,
        base_gpu_id: int = 0,
        gpu_id_step: int = 1,
        gpu_ids: Optional[List[int]] = None,  # 直接指定GPU ID列表
        
        # 内存和批处理配置
        mem_fraction_static: float = 0.9,
        max_running_requests: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        chunked_prefill_size: Optional[int] = None,
        
        # 模型配置
        dtype: str = "auto",  # auto, half, float16, bfloat16, float, float32
        quantization: Optional[str] = None,  # awq, fp8, gptq, etc.
        kv_cache_dtype: str = "auto",  # auto, fp8_e5m2, fp8_e4m3
        context_length: Optional[int] = None,
        
        # 优化选项
        enable_torch_compile: bool = False,
        disable_cuda_graph: bool = False,
        attention_backend: str = "torch_native",  # torch_native, flashinfer, triton (默认使用torch_native避免编译问题)
        
        # 其他配置
        trust_remote_code: bool = False,
        random_seed: Optional[int] = None,
        log_level: str = "INFO",
        
        # 高级配置
        enable_prefix_caching: bool = True,
        disable_radix_cache: bool = False,
        enable_mixed_precision: bool = True
    ):
        """
        初始化SGLang引擎
        
        Args:
            model_path: 模型路径 (本地路径或HuggingFace模型ID)
            device: 设备类型 (cuda, cpu, xpu, hpu)
            tp_size: 张量并行大小
            dp_size: 数据并行大小
            base_gpu_id: 起始GPU ID
            gpu_id_step: GPU ID步长
            gpu_ids: 直接指定GPU ID列表，优先级高于base_gpu_id和gpu_id_step
            mem_fraction_static: 静态内存分配比例
            max_running_requests: 最大并发请求数
            max_total_tokens: 最大总token数
            chunked_prefill_size: 分块预填充大小
            dtype: 数据类型
            quantization: 量化方法
            kv_cache_dtype: KV缓存数据类型
            context_length: 上下文长度
            enable_torch_compile: 启用torch.compile加速
            disable_cuda_graph: 禁用CUDA图优化
            attention_backend: 注意力后端
            trust_remote_code: 信任远程代码
            random_seed: 随机种子
            log_level: 日志级别
            enable_prefix_caching: 启用前缀缓存
            disable_radix_cache: 禁用RadixAttention缓存
            enable_mixed_precision: 启用混合精度
        """
        self.model_path = model_path
        self.device = device
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.base_gpu_id = base_gpu_id
        self.gpu_id_step = gpu_id_step
        self.gpu_ids = gpu_ids
        
        # 设置日志
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)
        
        # GPU设备管理
        self._setup_gpu_config()
        
        # 构建引擎参数
        self.engine_args = {
            "model_path": model_path,
            "device": device,
            "tp_size": tp_size,
            "dp_size": dp_size,
            "mem_fraction_static": mem_fraction_static,
            "dtype": dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "trust_remote_code": trust_remote_code,
            "attention_backend": attention_backend,
        }
        
        # 添加可选参数
        if max_running_requests is not None:
            self.engine_args["max_running_requests"] = max_running_requests
        if max_total_tokens is not None:
            self.engine_args["max_total_tokens"] = max_total_tokens
        if chunked_prefill_size is not None:
            self.engine_args["chunked_prefill_size"] = chunked_prefill_size
        if quantization is not None:
            self.engine_args["quantization"] = quantization
        if context_length is not None:
            self.engine_args["context_length"] = context_length
        if random_seed is not None:
            self.engine_args["random_seed"] = random_seed
        if base_gpu_id != 0:
            self.engine_args["base_gpu_id"] = base_gpu_id
        if gpu_id_step != 1:
            self.engine_args["gpu_id_step"] = gpu_id_step
        if enable_torch_compile:
            self.engine_args["enable_torch_compile"] = True
        if disable_cuda_graph:
            self.engine_args["disable_cuda_graph"] = True
        if not enable_prefix_caching:
            self.engine_args["disable_prefix_caching"] = True
        if disable_radix_cache:
            self.engine_args["disable_radix_cache"] = True
            
        self.llm = None
        self.logger.info(f"SGLangEngine initialized with args: {self.engine_args}")
    
    def _setup_gpu_config(self):
        """设置GPU配置"""
        if self.device == "cuda":
            # 检查CUDA可用性
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                return
            
            # 获取原始可用GPU数量（不受CUDA_VISIBLE_DEVICES影响）
            # 如果CUDA_VISIBLE_DEVICES已设置，先临时清除以获取真实GPU数量
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if original_cuda_visible is not None:
                # 临时清除CUDA_VISIBLE_DEVICES以获取真实GPU数量
                del os.environ["CUDA_VISIBLE_DEVICES"]
                # 重新初始化CUDA以获取真实GPU数量
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
            
            available_gpus = torch.cuda.device_count()
            self.logger.info(f"Total available GPUs: {available_gpus}")
            
            # 设置GPU ID列表
            if self.gpu_ids is not None:
                # 验证指定的GPU ID
                for gpu_id in self.gpu_ids:
                    if gpu_id >= available_gpus:
                        raise ValueError(f"GPU {gpu_id} not available. Only {available_gpus} GPUs found.")
                self.logger.info(f"Using specified GPUs: {self.gpu_ids}")
            else:
                # 根据base_gpu_id和tp_size计算GPU列表
                required_gpus = self.base_gpu_id + self.tp_size * self.gpu_id_step
                if required_gpus > available_gpus:
                    raise ValueError(f"Required {required_gpus} GPUs but only {available_gpus} available")
                
                self.gpu_ids = [self.base_gpu_id + i * self.gpu_id_step for i in range(self.tp_size)]
                self.logger.info(f"Calculated GPU IDs: {self.gpu_ids}")
            
            # 设置CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            self.logger.info(f"Set CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            
            # 重新初始化CUDA以应用新的CUDA_VISIBLE_DEVICES设置
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"device": self.device, "gpu_count": 0}
        
        gpu_info = {
            "device": self.device,
            "gpu_count": len(self.gpu_ids) if self.gpu_ids else 0,
            "gpu_ids": self.gpu_ids,
            "total_memory": [],
            "allocated_memory": [],
            "cached_memory": []
        }
        
        for gpu_id in (self.gpu_ids or []):
            if gpu_id < torch.cuda.device_count():
                gpu_info["total_memory"].append(torch.cuda.get_device_properties(gpu_id).total_memory)
                gpu_info["allocated_memory"].append(torch.cuda.memory_allocated(gpu_id))
                gpu_info["cached_memory"].append(torch.cuda.memory_reserved(gpu_id))
        
        return gpu_info
    
    def update_gpu_config(self, gpu_ids: List[int]):
        """动态更新GPU配置"""
        if self.llm is not None:
            raise RuntimeError("Cannot update GPU config while engine is running. Shutdown first.")
        
        # 清除当前的CUDA_VISIBLE_DEVICES设置以允许访问所有GPU
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
            self.logger.info("Cleared CUDA_VISIBLE_DEVICES for GPU config update")
        
        # 清理CUDA缓存
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        
        self.gpu_ids = gpu_ids
        self.tp_size = len(gpu_ids)
        self.engine_args["tp_size"] = self.tp_size
        self._setup_gpu_config()
        self.logger.info(f"Updated GPU config: {gpu_ids}")
    
    def start_engine(self):
        """启动推理引擎"""
        if self.llm is not None:
            self.logger.warning("Engine already started")
            return
            
        try:
            self.llm = sgl.Engine(**self.engine_args)
            self.logger.info("SGLang engine started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start engine: {e}")
            raise
    
    def shutdown(self):
        """关闭推理引擎"""
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None
            self.logger.info("SGLang engine shutdown")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
    
    def generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        同步生成 (非流式)
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Returns:
            包含生成文本和元数据的字典
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        result = self.llm.generate(prompt, sampling_params)
        return result if isinstance(result, dict) else {"text": result}
    
    def generate_text_only(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        同步生成 (非流式) - 仅返回文本
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Returns:
            生成的文本
        """
        result = self.generate(prompt, sampling_params)
        return result.get("text", result) if isinstance(result, dict) else str(result)
    
    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        同步流式生成 (使用stream_and_merge)
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Returns:
            完整的生成文本
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        return stream_and_merge(self.llm, prompt, sampling_params)
    
    def generate_stream_iter(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        同步流式生成 (返回迭代器)
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Yields:
            生成的文本块
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        # 直接使用SGLang的同步流式生成
        try:
            # 尝试使用SGLang的内置流式生成方法
            if hasattr(self.llm, 'generate_stream'):
                for chunk in self.llm.generate_stream(prompt, sampling_params):
                    if isinstance(chunk, dict):
                        yield chunk.get('text', str(chunk))
                    else:
                        yield str(chunk)
            else:
                # 如果没有流式方法，使用stream_and_merge的分块方式
                from sglang.utils import stream_and_merge
                full_text = stream_and_merge(self.llm, prompt, sampling_params)
                # 简单地按句子分块输出
                sentences = full_text.split('. ')
                for i, sentence in enumerate(sentences):
                    if i < len(sentences) - 1:
                        yield sentence + '. '
                    else:
                        yield sentence
        except Exception as e:
            # 如果流式方式失败，使用简单的同步方式
            result = self.generate_text_only(prompt, sampling_params)
            yield result
    
    async def async_generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        异步生成 (支持流式和非流式)
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            stream: 是否使用流式生成
            
        Returns:
            非流式: 包含生成文本和元数据的字典
            流式: 异步生成器
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        if stream:
            # 对于流式生成，使用专门的流式API
            return self.async_generate_stream_raw(prompt, sampling_params)
        else:
            # 对于非流式生成，不传递stream参数
            return await self.llm.async_generate(prompt, sampling_params)
    
    async def async_generate_text_only(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        异步生成 (非流式) - 仅返回文本
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Returns:
            生成的文本
        """
        result = await self.async_generate(prompt, sampling_params, stream=False)
        return result.get("text", result) if isinstance(result, dict) else str(result)
    
    async def async_generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        异步流式生成 (使用async_stream_and_merge)
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Yields:
            生成的文本块
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        async for chunk in async_stream_and_merge(self.llm, prompt, sampling_params):
            yield chunk
    
    async def async_generate_stream_raw(
        self,
        prompt: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        异步流式生成 (原始输出，包含元数据)
        
        Args:
            prompt: 输入提示
            sampling_params: 采样参数
            
        Yields:
            包含文本和元数据的字典
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        # 使用SGLang的异步流式生成API
        try:
            # 尝试使用SGLang的内置流式方法
            async for chunk in self.llm.async_generate_stream(prompt, sampling_params):
                yield chunk
        except AttributeError:
            # 如果没有async_generate_stream方法，使用async_stream_and_merge
            async for chunk_text in async_stream_and_merge(self.llm, prompt, sampling_params):
                yield {"text": chunk_text}
    
    def batch_generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量生成 (非流式)
        
        Args:
            prompts: 输入提示列表
            sampling_params: 采样参数
            
        Returns:
            生成结果列表 (包含文本和元数据)
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        # 使用SGLang的批量生成
        results = self.llm.generate(prompts, sampling_params)
        return results if isinstance(results, list) else [results]
    
    def batch_generate_text_only(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        批量生成 (非流式) - 仅返回文本
        
        Args:
            prompts: 输入提示列表
            sampling_params: 采样参数
            
        Returns:
            生成的文本列表
        """
        results = self.batch_generate(prompts, sampling_params)
        return [result.get("text", result) if isinstance(result, dict) else str(result) for result in results]
    
    async def async_batch_generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        异步批量生成 (非流式)
        
        Args:
            prompts: 输入提示列表
            sampling_params: 采样参数
            
        Returns:
            生成结果列表
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        # 并发执行异步生成
        tasks = [self.async_generate(prompt, sampling_params, stream=False) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    async def async_batch_generate_text_only(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        异步批量生成 (非流式) - 仅返回文本
        
        Args:
            prompts: 输入提示列表
            sampling_params: 采样参数
            
        Returns:
            生成的文本列表
        """
        results = await self.async_batch_generate(prompts, sampling_params)
        return [result.get("text", result) if isinstance(result, dict) else str(result) for result in results]
    
    async def async_batch_generate_stream(
        self,
        prompts: List[str],
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[AsyncGenerator[str, None]]:
        """
        异步批量流式生成
        
        Args:
            prompts: 输入提示列表
            sampling_params: 采样参数
            
        Returns:
            异步生成器列表
        """
        if self.llm is None:
            raise RuntimeError("Engine not started. Call start_engine() first.")
            
        if sampling_params is None:
            sampling_params = {"temperature": 0.8, "top_p": 0.95}
            
        generators = []
        for prompt in prompts:
            generators.append(self.async_generate_stream(prompt, sampling_params))
        return generators
    
    # 便捷方法：统一推理接口
    def infer(
        self,
        prompt: Union[str, List[str]],
        sampling_params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        async_mode: bool = False,
        text_only: bool = True
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]], AsyncGenerator, List[AsyncGenerator]]:
        """
        统一推理接口
        
        Args:
            prompt: 单个提示或提示列表
            sampling_params: 采样参数
            stream: 是否使用流式生成
            async_mode: 是否使用异步模式
            text_only: 是否仅返回文本
            
        Returns:
            根据参数返回不同格式的结果
        """
        if async_mode:
            if isinstance(prompt, list):
                if stream:
                    return self.async_batch_generate_stream(prompt, sampling_params)
                else:
                    return self.async_batch_generate_text_only(prompt, sampling_params) if text_only else self.async_batch_generate(prompt, sampling_params)
            else:
                if stream:
                    return self.async_generate_stream(prompt, sampling_params)
                else:
                    return self.async_generate_text_only(prompt, sampling_params) if text_only else self.async_generate(prompt, sampling_params, stream=False)
        else:
            if isinstance(prompt, list):
                return self.batch_generate_text_only(prompt, sampling_params) if text_only else self.batch_generate(prompt, sampling_params)
            else:
                if stream:
                    return self.generate_stream(prompt, sampling_params) if text_only else self.generate_stream_iter(prompt, sampling_params)
                else:
                    return self.generate_text_only(prompt, sampling_params) if text_only else self.generate(prompt, sampling_params)


# 便捷函数
def create_engine(
    model_path: str,
    gpu_ids: Optional[List[int]] = None,
    base_gpu_id: int = 0,
    gpu_id_step: int = 1,
    tp_size: int = 1,
    batch_size_config: Optional[Dict[str, int]] = None,
    **kwargs
) -> SGLangEngine:
    """
    便捷的引擎创建函数
    
    Args:
        model_path: 模型路径
        gpu_ids: 直接指定GPU ID列表
        base_gpu_id: 起始GPU ID (当gpu_ids为None时使用)
        gpu_id_step: GPU ID步长
        tp_size: 张量并行大小
        batch_size_config: 批处理配置 {"max_running_requests": int, "max_total_tokens": int}
        **kwargs: 其他参数
        
    Returns:
        SGLangEngine实例
    """
    engine_kwargs = {
        "model_path": model_path,
        "tp_size": tp_size,
        "base_gpu_id": base_gpu_id,
        "gpu_id_step": gpu_id_step,
        **kwargs
    }
    
    if gpu_ids is not None:
        engine_kwargs["gpu_ids"] = gpu_ids
        engine_kwargs["tp_size"] = len(gpu_ids)  # 自动设置tp_size
    
    if batch_size_config:
        engine_kwargs.update(batch_size_config)
    
    return SGLangEngine(**engine_kwargs)

def create_engine_with_gpus(*gpu_ids: int, model_path: str, **kwargs) -> SGLangEngine:
    """
    使用指定GPU创建引擎的便捷函数
    
    Args:
        *gpu_ids: GPU ID列表
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        SGLangEngine实例
        
    Example:
        engine = create_engine_with_gpus(0, 1, 2, model_path="/path/to/model")
    """
    return create_engine(model_path=model_path, gpu_ids=list(gpu_ids), **kwargs)

def get_default_sampling_params(
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = -1,
    max_new_tokens: int = 512,
    stop: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    获取默认采样参数
    
    Args:
        temperature: 温度参数
        top_p: top-p采样参数
        top_k: top-k采样参数
        max_new_tokens: 最大新token数
        stop: 停止词
        **kwargs: 其他参数
        
    Returns:
        采样参数字典
    """
    params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        **kwargs
    }
    
    if top_k > 0:
        params["top_k"] = top_k
    if stop is not None:
        params["stop"] = stop
        
    return params


if __name__ == '__main__':
    # 示例用法
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France's capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]
    
    # 使用便捷函数创建采样参数
    sampling_params = get_default_sampling_params(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=256,
        stop=["\n\n", "</s>"]
    )
    
    print("=== SGLang引擎功能演示 ===")
    
    # 方法1: 使用指定GPU创建引擎
    with create_engine_with_gpus(
        3, 4,  # 使用GPU 3和4
        model_path="/home/nfs02/model/Qwen_Qwen2.5-7B-Instruct",
        mem_fraction_static=0.7,
        quantization=None,  # 可选: "awq", "fp8", "gptq"
        enable_torch_compile=False, 
        attention_backend="torch_native", # 使用torch_native避免flashinfer编译问题
        kv_cache_dtype="auto"  # 避免FP8相关问题
    ) as engine:
        
        # 显示GPU信息
        gpu_info = engine.get_gpu_info()
        print(f"\n当前GPU配置: {gpu_info}")
        
        print("\n=== 1. 同步非流式生成 ===")
        result = engine.generate_text_only(prompts[0], sampling_params)
        print(f"提示: {prompts[0]}")
        print(f"结果: {result}")
        
        print("\n=== 2. 同步流式生成 ===")
        print(f"提示: {prompts[1]}")
        print("生成文本: ", end="", flush=True)
        for chunk in engine.generate_stream_iter(prompts[1], sampling_params):
            print(chunk, end="", flush=True)
        print()  # 换行
        
        print("\n=== 3. 批量生成 ===")
        batch_results = engine.batch_generate_text_only(prompts, sampling_params)
        for i, result in enumerate(batch_results):
            print(f"\n提示 {i+1}: {prompts[i][:50]}...")
            print(f"结果: {result[:100]}...")
        
        print("\n=== 4. 异步流式生成 ===")
        async def test_async_streaming():
            print(f"提示: {prompts[2]}")
            print("生成文本: ", end="", flush=True)
            
            async for chunk in engine.async_generate_stream(prompts[2], sampling_params):
                print(chunk, end="", flush=True)
            
            print()  # 换行
        
        asyncio.run(test_async_streaming())
        
        print("\n=== 5. 异步批量生成 ===")
        async def test_async_batch():
            results = await engine.async_batch_generate_text_only(prompts[:2], sampling_params)
            for i, result in enumerate(results):
                print(f"\n异步结果 {i+1}: {result[:100]}...")
        
        asyncio.run(test_async_batch())
        
        print("\n=== 6. 统一推理接口演示 ===")
        # 同步单个
        sync_result = engine.infer(prompts[0], sampling_params, stream=False, async_mode=False, text_only=True)
        print(f"同步单个: {sync_result}")
        
        # 同步批量
        sync_batch = engine.infer(prompts[:2], sampling_params, stream=False, async_mode=False, text_only=True)
        print(f"同步批量: {len(sync_batch)} 个结果")
        for i, result in enumerate(sync_batch):
            print(f"  结果 {i+1}: {result}")
        
        # 异步单个
        async def test_unified_async():
            async_result = await engine.infer(prompts[0], sampling_params, stream=False, async_mode=True, text_only=True)
            print(f"异步单个: {async_result}")
        
        asyncio.run(test_unified_async())

        async def test_unified_async():
            async_result = await engine.infer(prompts[:2], sampling_params, stream=False, async_mode=True, text_only=True)
            print(f"异步批量: {async_result}")
        
        asyncio.run(test_unified_async())
    
    print("\n=== 7. 动态GPU配置演示 ===")
    # 方法2: 先创建引擎，后配置GPU
    engine2 = create_engine(
        model_path="/home/nfs02/model/Qwen_Qwen2.5-7B-Instruct",
        gpu_ids=[5,6],  # 初始使用GPU 0
        mem_fraction_static=0.6
    )
    
    print(f"初始GPU配置: {engine2.gpu_ids}")
    
    # 动态更新GPU配置
    try:
        engine2.update_gpu_config([3, 4])  # 切换到GPU 1和2
        print(f"更新后GPU配置: {engine2.gpu_ids}")
        
        with engine2:
            result = engine2.generate_text_only("Hello, world!", sampling_params)
            print(f"使用新GPU配置的生成结果: {result[:50]}...")
    except Exception as e:
        print(f"GPU配置更新失败: {e}")
    
    print("\n=== 演示完成 ===")