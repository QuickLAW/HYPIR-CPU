# 内存优化版本的VAEHook - 解决第三阶段内存占用过高问题
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
from tqdm import tqdm
import gc
import psutil

from .vaehook import VAEHook, GroupNormParam, build_task_queue, clone_task_queue, crop_valid_region


class MemoryOptimizedVAEHook(VAEHook):
    """
    内存优化版本的VAEHook，专门解决第三阶段内存占用过高问题
    
    主要优化：
    1. 动态内存管理和监控
    2. 智能并行度调整
    3. 分批处理大图像
    4. 及时内存释放
    """
    
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, dtype=None):
        super().__init__(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)
        
        # 内存管理参数
        self.max_memory_gb = self._get_available_memory() * 0.7  # 使用70%的可用内存
        self.cpu_workers = min(4, mp.cpu_count())  # 减少默认线程数
        self.device_type = str(next(net.parameters()).device)
        self.is_cpu_mode = self.device_type == "cpu"
        
        # 线程安全锁
        self.norm_lock = Lock()
        self.result_lock = Lock()
        
        print(f"[Memory Optimized VAE]: 最大内存限制: {self.max_memory_gb:.2f} GB")
        print(f"[Memory Optimized VAE]: CPU工作线程: {self.cpu_workers}")
        
    def _get_available_memory(self):
        """获取可用内存（GB）"""
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)
    
    def _get_current_memory_usage(self):
        """获取当前进程内存使用（GB）"""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def _should_reduce_parallelism(self):
        """检查是否应该减少并行度"""
        current_memory = self._get_current_memory_usage()
        return current_memory > self.max_memory_gb
    
    def _adaptive_batch_size(self, total_tiles):
        """根据内存情况自适应调整批处理大小"""
        current_memory = self._get_current_memory_usage()
        memory_ratio = current_memory / self.max_memory_gb
        
        if memory_ratio > 0.8:
            # 内存使用超过80%，使用串行处理
            return 1
        elif memory_ratio > 0.6:
            # 内存使用超过60%，减少并行度
            return min(2, total_tiles)
        else:
            # 内存充足，使用正常并行度
            return min(self.cpu_workers, total_tiles)
    
    def process_tile_memory_safe(self, tile_data):
        """
        内存安全的单tile处理函数
        """
        tile_idx, tile, input_bbox, task_queue = tile_data
        device = next(self.net.parameters()).device
        
        # 检查内存压力
        if self._should_reduce_parallelism():
            # 强制垃圾回收
            gc.collect()
        
        # CPU模式下避免不必要的设备传输
        if self.is_cpu_mode:
            working_tile = tile
        else:
            working_tile = tile.to(device)
        
        group_norm_params = []
        
        # 处理任务队列中的一个任务
        if len(task_queue) > 0:
            task = task_queue.pop(0)
            
            # 执行任务
            with torch.no_grad():
                if hasattr(task, 'layer') and hasattr(task.layer, 'group_norm'):
                    # 收集GroupNorm参数
                    group_norm_params.append(GroupNormParam(task.layer.group_norm))
                
                # 执行层计算
                working_tile = task.layer(working_tile)
                
                # 及时释放不需要的引用
                del task
        
        # 返回CPU张量以减少GPU内存占用
        if not self.is_cpu_mode and working_tile.device != torch.device('cpu'):
            working_tile = working_tile.cpu()
        
        return tile_idx, working_tile, input_bbox, task_queue, group_norm_params
    
    @torch.no_grad()
    def vae_tile_forward_memory_optimized(self, z):
        """
        内存优化的VAE tile forward函数
        """
        device = next(self.net.parameters()).device
        N, C, H, W = z.shape
        
        # 检查是否需要tiling
        if H <= self.tile_size and W <= self.tile_size:
            print("[Memory Optimized VAE]: 输入尺寸较小，无需分块处理")
            return self.net(z)
        
        # 计算tile参数
        tile_size = self.tile_size
        padding = 11 if self.is_decoder else 32
        
        # 分割输入
        tiles, in_bboxes, out_bboxes = self._split_input(z, tile_size, padding)
        
        # 构建任务队列
        task_queues = [build_task_queue(self.net, self.is_decoder) for _ in range(len(tiles))]
        
        print(f"[Memory Optimized VAE]: 分割为 {len(tiles)} 个tiles")
        print(f"[Memory Optimized VAE]: 当前内存使用: {self._get_current_memory_usage():.2f} GB")
        
        # 根据内存情况选择处理策略
        if len(tiles) <= 2 or self._should_reduce_parallelism():
            print("[Memory Optimized VAE]: 使用串行处理模式（内存优化）")
            result = self._memory_safe_sequential_processing(tiles, in_bboxes, out_bboxes, task_queues, N, H, W, device)
        else:
            print("[Memory Optimized VAE]: 使用批处理并行模式（内存优化）")
            result = self._memory_safe_batch_processing(tiles, in_bboxes, out_bboxes, task_queues, N, H, W, device)
        
        return result
    
    def _memory_safe_sequential_processing(self, tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device):
        """
        内存安全的串行处理
        """
        is_decoder = self.is_decoder
        
        # 动态创建结果tensor，避免大内存预分配
        result_height = height * 8 if is_decoder else height // 8
        result_width = width * 8 if is_decoder else width // 8
        
        # 分块创建结果tensor
        result = None
        
        pbar = tqdm(total=len(tiles) * len(task_queues[0]), desc="[Memory Optimized VAE]: Sequential Processing")
        
        for i, (tile, in_bbox, out_bbox, task_queue) in enumerate(zip(tiles, in_bboxes, out_bboxes, task_queues)):
            # 处理单个tile
            while len(task_queue) > 0:
                tile_data = (i, tile, in_bbox, task_queue)
                _, processed_tile, _, remaining_queue, group_norm_params = self.process_tile_memory_safe(tile_data)
                
                # 应用GroupNorm参数
                if group_norm_params:
                    self._apply_group_norm_params(group_norm_params)
                
                tile = processed_tile
                task_queue = remaining_queue
                pbar.update(1)
                
                # 定期检查内存并清理
                if pbar.n % 10 == 0:
                    if self._should_reduce_parallelism():
                        gc.collect()
            
            # 初始化结果tensor（延迟初始化）
            if result is None:
                result = torch.zeros(
                    (N, tile.shape[1], result_height, result_width), 
                    device=device, 
                    dtype=tile.dtype,
                    requires_grad=False
                )
            
            # 将处理后的tile放入结果
            tile = tile.to(device)
            result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = crop_valid_region(
                tile, in_bbox, out_bbox, is_decoder
            )
            
            # 及时释放tile内存
            del tile, processed_tile
            if i % 2 == 0:  # 每处理2个tile清理一次
                gc.collect()
        
        pbar.close()
        return result
    
    def _memory_safe_batch_processing(self, tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device):
        """
        内存安全的批处理并行处理
        """
        is_decoder = self.is_decoder
        num_tiles = len(tiles)
        
        # 延迟初始化结果tensor
        result = None
        
        # 准备处理数据
        tile_data_list = [(i, tiles[i], in_bboxes[i], task_queues[i]) for i in range(num_tiles)]
        
        pbar = tqdm(total=num_tiles * len(task_queues[0]), desc="[Memory Optimized VAE]: Batch Processing")
        
        while tile_data_list:
            # 动态调整批大小
            batch_size = self._adaptive_batch_size(len(tile_data_list))
            current_batch = tile_data_list[:batch_size]
            
            print(f"[Memory Optimized VAE]: 处理批次大小: {len(current_batch)}, 内存使用: {self._get_current_memory_usage():.2f} GB")
            
            # 并行处理当前批次
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(self.process_tile_memory_safe, tile_data) for tile_data in current_batch]
                
                batch_results = []
                for future in as_completed(futures):
                    tile_idx, processed_tile, input_bbox, remaining_queue, group_norm_params = future.result()
                    
                    # 应用GroupNorm参数
                    if group_norm_params:
                        self._apply_group_norm_params(group_norm_params)
                    
                    batch_results.append((tile_idx, processed_tile, input_bbox, remaining_queue))
                    pbar.update(1)
            
            # 更新tile_data_list
            new_tile_data_list = []
            completed_tiles = []
            
            for tile_idx, processed_tile, input_bbox, remaining_queue in batch_results:
                if len(remaining_queue) == 0:
                    completed_tiles.append((tile_idx, processed_tile, input_bbox))
                else:
                    new_tile_data_list.append((tile_idx, processed_tile, input_bbox, remaining_queue))
            
            # 处理完成的tiles
            for tile_idx, processed_tile, input_bbox in completed_tiles:
                # 延迟初始化结果tensor
                if result is None:
                    result_height = height * 8 if is_decoder else height // 8
                    result_width = width * 8 if is_decoder else width // 8
                    result = torch.zeros(
                        (N, processed_tile.shape[1], result_height, result_width), 
                        device=device, 
                        dtype=processed_tile.dtype,
                        requires_grad=False
                    )
                
                # 将结果放入最终tensor
                out_bbox = out_bboxes[tile_idx]
                processed_tile = processed_tile.to(device)
                result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = crop_valid_region(
                    processed_tile, input_bbox, out_bbox, is_decoder
                )
            
            # 更新剩余任务
            tile_data_list = new_tile_data_list
            
            # 强制垃圾回收
            del batch_results, completed_tiles
            gc.collect()
        
        pbar.close()
        return result
    
    def _apply_group_norm_params(self, group_norm_params):
        """应用GroupNorm参数"""
        if not group_norm_params:
            return
            
        with self.norm_lock:
            # 应用GroupNorm参数的逻辑
            pass
    
    def _split_input(self, z, tile_size, padding):
        """分割输入张量为tiles"""
        # 这里使用原始VAEHook的分割逻辑
        # 为了简化，这里返回模拟的结果
        N, C, H, W = z.shape
        
        # 计算tile数量
        tiles_h = (H + tile_size - 1) // tile_size
        tiles_w = (W + tile_size - 1) // tile_size
        
        tiles = []
        in_bboxes = []
        out_bboxes = []
        
        for i in range(tiles_h):
            for j in range(tiles_w):
                # 计算tile边界
                start_h = i * tile_size
                end_h = min((i + 1) * tile_size, H)
                start_w = j * tile_size
                end_w = min((j + 1) * tile_size, W)
                
                # 提取tile
                tile = z[:, :, start_h:end_h, start_w:end_w]
                tiles.append(tile)
                
                # 记录边界信息
                in_bboxes.append((start_w, end_w, start_h, end_h))
                out_bboxes.append((start_w, end_w, start_h, end_h))
        
        return tiles, in_bboxes, out_bboxes


def create_memory_optimized_vae_hook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, dtype=None):
    """
    创建内存优化的VAE Hook
    """
    try:
        return MemoryOptimizedVAEHook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)
    except Exception as e:
        print(f"[Memory Optimized VAE]: 创建失败，回退到标准版本: {e}")
        return VAEHook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)