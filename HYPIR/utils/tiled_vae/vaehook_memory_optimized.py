# Memory-optimized VAEHook for stage-3 memory pressure
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
    Memory-optimized VAEHook to address high memory usage during stage-3 processing.

    Optimizations:
    1. Dynamic memory monitoring and management
    2. Adaptive parallelism
    3. Batched processing for large images
    4. Proactive memory reclamation
    """
    
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, dtype=None):
        super().__init__(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)
        
        # Memory budget and worker limits
        self.max_memory_gb = self._get_available_memory() * 0.7
        self.cpu_workers = min(4, mp.cpu_count())
        self.device_type = str(next(net.parameters()).device)
        self.is_cpu_mode = self.device_type == "cpu"
        
        # Thread-safety locks
        self.norm_lock = Lock()
        self.result_lock = Lock()
        
        print(f"[Memory Optimized VAE]: Memory limit: {self.max_memory_gb:.2f} GB")
        print(f"[Memory Optimized VAE]: CPU workers: {self.cpu_workers}")
        
    def _get_available_memory(self):
        """Get available system memory in GB."""
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)
    
    def _get_current_memory_usage(self):
        """Get current process memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    
    def _should_reduce_parallelism(self):
        """Check whether to reduce parallelism."""
        current_memory = self._get_current_memory_usage()
        return current_memory > self.max_memory_gb
    
    def _adaptive_batch_size(self, total_tiles):
        """Adapt batch size based on memory pressure."""
        current_memory = self._get_current_memory_usage()
        memory_ratio = current_memory / self.max_memory_gb
        
        if memory_ratio > 0.8:
            # High memory pressure: serialize
            return 1
        elif memory_ratio > 0.6:
            # Moderate pressure: reduce parallelism
            return min(2, total_tiles)
        else:
            # Sufficient memory: normal parallelism
            return min(self.cpu_workers, total_tiles)
    
    def process_tile_memory_safe(self, tile_data):
        """
        Memory-safe single-tile processing.
        """
        tile_idx, tile, input_bbox, task_queue = tile_data
        device = next(self.net.parameters()).device
        
        # Check memory pressure
        if self._should_reduce_parallelism():
            # Force garbage collection
            gc.collect()
        
        # Avoid unnecessary transfers in CPU mode
        if self.is_cpu_mode:
            working_tile = tile
        else:
            working_tile = tile.to(device)
        
        group_norm_params = []
        
        # Process one task from the queue
        if len(task_queue) > 0:
            task = task_queue.pop(0)
            
            # Execute task
            with torch.no_grad():
                if hasattr(task, 'layer') and hasattr(task.layer, 'group_norm'):
                    # Collect GroupNorm params
                    group_norm_params.append(GroupNormParam(task.layer.group_norm))
                
                # Run layer computation
                working_tile = task.layer(working_tile)
                
                # Release references early
                del task
        
        # Return CPU tensor to reduce GPU pressure
        if not self.is_cpu_mode and working_tile.device != torch.device('cpu'):
            working_tile = working_tile.cpu()
        
        return tile_idx, working_tile, input_bbox, task_queue, group_norm_params
    
    @torch.no_grad()
    def vae_tile_forward_memory_optimized(self, z):
        """
        Memory-optimized VAE tile forward.
        """
        device = next(self.net.parameters()).device
        N, C, H, W = z.shape
        
        # Skip tiling for small inputs
        if H <= self.tile_size and W <= self.tile_size:
            print("[Memory Optimized VAE]: Input is small; tiling is not required")
            return self.net(z)
        
        # Compute tile parameters
        tile_size = self.tile_size
        padding = 11 if self.is_decoder else 32
        
        # Split input
        tiles, in_bboxes, out_bboxes = self._split_input(z, tile_size, padding)
        
        # Build task queues
        task_queues = [build_task_queue(self.net, self.is_decoder) for _ in range(len(tiles))]
        
        print(f"[Memory Optimized VAE]: Split into {len(tiles)} tiles")
        print(f"[Memory Optimized VAE]: Current memory usage: {self._get_current_memory_usage():.2f} GB")
        
        # Choose strategy based on memory pressure
        if len(tiles) <= 2 or self._should_reduce_parallelism():
            print("[Memory Optimized VAE]: Using sequential mode (memory-optimized)")
            result = self._memory_safe_sequential_processing(tiles, in_bboxes, out_bboxes, task_queues, N, H, W, device)
        else:
            print("[Memory Optimized VAE]: Using batched parallel mode (memory-optimized)")
            result = self._memory_safe_batch_processing(tiles, in_bboxes, out_bboxes, task_queues, N, H, W, device)
        
        return result
    
    def _memory_safe_sequential_processing(self, tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device):
        """
        Memory-safe sequential processing.
        """
        is_decoder = self.is_decoder
        
        # Build output tensor lazily to avoid large pre-allocation
        result_height = height * 8 if is_decoder else height // 8
        result_width = width * 8 if is_decoder else width // 8
        
        # Lazy init
        result = None
        
        pbar = tqdm(total=len(tiles) * len(task_queues[0]), desc="[Memory Optimized VAE]: Sequential Processing")
        
        for i, (tile, in_bbox, out_bbox, task_queue) in enumerate(zip(tiles, in_bboxes, out_bboxes, task_queues)):
            # Process a single tile
            while len(task_queue) > 0:
                tile_data = (i, tile, in_bbox, task_queue)
                _, processed_tile, _, remaining_queue, group_norm_params = self.process_tile_memory_safe(tile_data)
                
                # Apply GroupNorm params
                if group_norm_params:
                    self._apply_group_norm_params(group_norm_params)
                
                tile = processed_tile
                task_queue = remaining_queue
                pbar.update(1)
                
                # Periodic memory check
                if pbar.n % 10 == 0:
                    if self._should_reduce_parallelism():
                        gc.collect()
            
            # Lazy init of output tensor
            if result is None:
                result = torch.zeros(
                    (N, tile.shape[1], result_height, result_width), 
                    device=device, 
                    dtype=tile.dtype,
                    requires_grad=False
                )
            
            # Write tile to output
            tile = tile.to(device)
            result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = crop_valid_region(
                tile, in_bbox, out_bbox, is_decoder
            )
            
            # Release tile memory
            del tile, processed_tile
            if i % 2 == 0:
                gc.collect()
        
        pbar.close()
        return result
    
    def _memory_safe_batch_processing(self, tiles, in_bboxes, out_bboxes, task_queues, N, height, width, device):
        """
        Memory-safe batched parallel processing.
        """
        is_decoder = self.is_decoder
        num_tiles = len(tiles)
        
        # Lazy init of output tensor
        result = None
        
        # Prepare work items
        tile_data_list = [(i, tiles[i], in_bboxes[i], task_queues[i]) for i in range(num_tiles)]
        
        pbar = tqdm(total=num_tiles * len(task_queues[0]), desc="[Memory Optimized VAE]: Batch Processing")
        
        while tile_data_list:
            # Adapt batch size
            batch_size = self._adaptive_batch_size(len(tile_data_list))
            current_batch = tile_data_list[:batch_size]
            
            print(f"[Memory Optimized VAE]: Batch size: {len(current_batch)}, memory usage: {self._get_current_memory_usage():.2f} GB")
            
            # Process current batch in parallel
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(self.process_tile_memory_safe, tile_data) for tile_data in current_batch]
                
                batch_results = []
                for future in as_completed(futures):
                    tile_idx, processed_tile, input_bbox, remaining_queue, group_norm_params = future.result()
                    
                    # Apply GroupNorm params
                    if group_norm_params:
                        self._apply_group_norm_params(group_norm_params)
                    
                    batch_results.append((tile_idx, processed_tile, input_bbox, remaining_queue))
                    pbar.update(1)
            
            # Refresh pending tiles
            new_tile_data_list = []
            completed_tiles = []
            
            for tile_idx, processed_tile, input_bbox, remaining_queue in batch_results:
                if len(remaining_queue) == 0:
                    completed_tiles.append((tile_idx, processed_tile, input_bbox))
                else:
                    new_tile_data_list.append((tile_idx, processed_tile, input_bbox, remaining_queue))
            
            # Completed tiles
            for tile_idx, processed_tile, input_bbox in completed_tiles:
                # Lazy init of output tensor
                if result is None:
                    result_height = height * 8 if is_decoder else height // 8
                    result_width = width * 8 if is_decoder else width // 8
                    result = torch.zeros(
                        (N, processed_tile.shape[1], result_height, result_width), 
                        device=device, 
                        dtype=processed_tile.dtype,
                        requires_grad=False
                    )
                
                # Write tile to output
                out_bbox = out_bboxes[tile_idx]
                processed_tile = processed_tile.to(device)
                result[:, :, out_bbox[2]:out_bbox[3], out_bbox[0]:out_bbox[1]] = crop_valid_region(
                    processed_tile, input_bbox, out_bbox, is_decoder
                )
            
            # Update remaining tasks
            tile_data_list = new_tile_data_list
            
            # Force garbage collection
            del batch_results, completed_tiles
            gc.collect()
        
        pbar.close()
        return result
    
    def _apply_group_norm_params(self, group_norm_params):
        """Apply GroupNorm parameters."""
        if not group_norm_params:
            return
            
        with self.norm_lock:
            # GroupNorm application placeholder
            pass
    
    def _split_input(self, z, tile_size, padding):
        """Split input tensor into tiles."""
        # Placeholder implementation
        N, C, H, W = z.shape
        
        # Compute tile grid
        tiles_h = (H + tile_size - 1) // tile_size
        tiles_w = (W + tile_size - 1) // tile_size
        
        tiles = []
        in_bboxes = []
        out_bboxes = []
        
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Compute tile bounds
                start_h = i * tile_size
                end_h = min((i + 1) * tile_size, H)
                start_w = j * tile_size
                end_w = min((j + 1) * tile_size, W)
                
                # Extract tile
                tile = z[:, :, start_h:end_h, start_w:end_w]
                tiles.append(tile)
                
                # Record bounds
                in_bboxes.append((start_w, end_w, start_h, end_h))
                out_bboxes.append((start_w, end_w, start_h, end_h))
        
        return tiles, in_bboxes, out_bboxes


def create_memory_optimized_vae_hook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, dtype=None):
    """
    Create a memory-optimized VAEHook.
    """
    try:
        return MemoryOptimizedVAEHook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)
    except Exception as e:
        print(f"[Memory Optimized VAE]: Creation failed; falling back to standard hook: {e}")
        return VAEHook(net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu, dtype)
