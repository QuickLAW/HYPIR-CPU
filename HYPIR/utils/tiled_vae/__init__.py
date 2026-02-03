from contextlib import contextmanager
from HYPIR.utils.tiled_vae.vaehook import VAEHook

# Try optimized hook; fall back to the base version if unavailable
try:
    from HYPIR.utils.tiled_vae.vaehook_optimized import create_optimized_vae_hook
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False


def enable_tiled_vae(
    vae,
    tile_size: int = 512,
    is_decoder: bool = True,
    fast_decoder: bool = True,
    fast_encoder: bool = True,
    color_fix: bool = False,
    to_gpu: bool = False,
    use_streaming: bool = True,
    use_memory_optimization: bool = True,
    use_optimization: bool = True,
    max_memory_gb: float = 4.0,
    progressive: bool = False,
    progress_callback=None,
    dtype=None
):
    """
    Enable tiled VAE processing with multiple optimization levels
    
    Args:
        vae: The VAE model to optimize
        tile_size: Size of tiles for processing
        is_decoder: Whether this is for decoder (True) or encoder (False)
        fast_decoder: Enable fast decoder optimizations
        fast_encoder: Enable fast encoder optimizations
        color_fix: Enable color correction
        to_gpu: Move processing to GPU
        use_streaming: Use streaming VAE (highest priority, solves memory issues)
        use_memory_optimization: Use memory-optimized version (fallback)
        use_optimization: Use performance-optimized version (fallback)
        max_memory_gb: Maximum memory limit for streaming mode (GB)
        progressive: Enable progressive processing with real-time updates
        progress_callback: Callback function for progress updates
        dtype: Data type for processing (for backward compatibility)
    
    Returns:
        Context manager for the optimized VAE hook
    """
    # Preserve backward compatibility
    return enable_tiled_vae_legacy(
        vae=vae,
        is_decoder=is_decoder,
        tile_size=tile_size,
        dtype=dtype,
        use_optimization=use_optimization,
        use_memory_optimization=use_memory_optimization
    )


@contextmanager
def enable_tiled_vae_legacy(
    vae,
    is_decoder,
    tile_size=256,
    dtype=None,
    use_optimization=True,
    use_memory_optimization=True,
):
    """
    Legacy context manager version for backward compatibility
    """
    if not is_decoder:
        original_forward = vae.encoder.forward
        
        # Resolve device type
        device_type = str(vae.device).split(':')[0] if hasattr(vae, 'device') else 'cpu'
        
        # Prefer memory-optimized hook when explicitly enabled on CPU
        if use_memory_optimization and device_type == 'cpu':
            try:
                from .vaehook_memory_optimized import MemoryOptimizedVAEHook
                print("Using Memory-Optimized VAEHook for enhanced memory efficiency")
                hook = MemoryOptimizedVAEHook(
                    vae.encoder, tile_size, is_decoder=False, 
                    fast_decoder=False, fast_encoder=True, 
                    color_fix=False, to_gpu=False, dtype=dtype
                )
            except ImportError:
                print("Memory-optimized VAEHook not available; falling back to optimized version")
                use_memory_optimization = False
        
        # If memory-optimized hook is unavailable, try the CPU-optimized hook
        if not use_memory_optimization and use_optimization and device_type == 'cpu':
            try:
                from .vaehook_optimized import OptimizedVAEHook
                print("Using Optimized VAEHook for enhanced CPU performance")
                hook = OptimizedVAEHook(
                    vae.encoder, tile_size, is_decoder=False, 
                    fast_decoder=False, fast_encoder=True, 
                    color_fix=False, to_gpu=False, dtype=dtype
                )
            except ImportError:
                print("Optimized VAEHook not available; using standard version")
                use_optimization = False
        
        # Fall back to the standard hook
        if not use_memory_optimization and not use_optimization:
            print("Using Standard VAEHook")
            hook = VAEHook(
                vae.encoder, tile_size, is_decoder=False, 
                fast_decoder=False, fast_encoder=True, 
                color_fix=False, to_gpu=False, dtype=dtype
            )
        
        vae.encoder.forward = hook
        vae.encoder.original_forward = original_forward
        try:
            yield
        finally:
            vae.encoder.forward = original_forward
            if hasattr(vae.encoder, "original_forward"):
                delattr(vae.encoder, "original_forward")
    else:
        original_forward = vae.decoder.forward
        
        # Resolve device type
        device_type = str(vae.device).split(':')[0] if hasattr(vae, 'device') else 'cpu'
        
        # Prefer memory-optimized hook when explicitly enabled on CPU
        if use_memory_optimization and device_type == 'cpu':
            try:
                from .vaehook_memory_optimized import MemoryOptimizedVAEHook
                print("Using Memory-Optimized VAEHook for enhanced memory efficiency")
                hook = MemoryOptimizedVAEHook(
                    vae.decoder, tile_size, is_decoder=True, 
                    fast_decoder=True, fast_encoder=False, 
                    color_fix=False, to_gpu=False, dtype=dtype
                )
            except ImportError:
                print("Memory-optimized VAEHook not available; falling back to optimized version")
                use_memory_optimization = False
        
        # If memory-optimized hook is unavailable, try the CPU-optimized hook
        if not use_memory_optimization and use_optimization and device_type == 'cpu':
            try:
                from .vaehook_optimized import OptimizedVAEHook
                print("Using Optimized VAEHook for enhanced CPU performance")
                hook = OptimizedVAEHook(
                    vae.decoder, tile_size, is_decoder=True, 
                    fast_decoder=True, fast_encoder=False, 
                    color_fix=False, to_gpu=False, dtype=dtype
                )
            except ImportError:
                print("Optimized VAEHook not available; using standard version")
                use_optimization = False
        
        # Fall back to the standard hook
        if not use_memory_optimization and not use_optimization:
            print("Using Standard VAEHook")
            hook = VAEHook(
                vae.decoder, tile_size, is_decoder=True, 
                fast_decoder=True, fast_encoder=False, 
                color_fix=False, to_gpu=False, dtype=dtype
            )
        
        vae.decoder.forward = hook
        vae.decoder.original_forward = original_forward
        try:
            yield
        finally:
            vae.decoder.forward = original_forward
            if hasattr(vae.decoder, "original_forward"):
                delattr(vae.decoder, "original_forward")
