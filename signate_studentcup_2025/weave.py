"""Weave initialization and configuration module.

This module provides utilities for initializing Weave tracing in the Wandb ecosystem.
"""

from loguru import logger
from typing import Any


# Weave initialization state
_weave_initialized = False


def init_weave(
    entity: str | None = None,
    project: str | None = None,
    enabled: bool = True,
) -> bool:
    """
    Initialize Weave for tracing.

    Args:
        entity: Wandb entity (username or team)
        project: Wandb project name
        enabled: Whether Weave is enabled

    Returns:
        True if Weave was initialized successfully, False otherwise
    """
    global _weave_initialized

    if not enabled:
        logger.info("Weave tracing is disabled")
        return False

    if _weave_initialized:
        logger.debug("Weave already initialized")
        return True

    try:
        import wandb

        # Check if wandb is initialized
        if wandb.run is None:
            logger.warning("Wandb run not initialized. Call wandb.init() before init_weave()")
            return False

        # Initialize Weave
        try:
            # Weave is part of wandb package in recent versions
            from wandb import weave as wandb_weave

            wandb_weave.init(
                project=project,
                entity=entity,
            )

            _weave_initialized = True
            logger.info(f"Weave tracing initialized: {entity}/{project}")
            return True

        except ImportError:
            # Fallback for older wandb versions or standalone weave
            try:
                import weave

                weave.init(
                    project_name=f"{entity}/{project}" if entity else project
                )

                _weave_initialized = True
                logger.info(f"Weave tracing initialized (standalone): {entity}/{project}")
                return True

            except ImportError:
                logger.warning("Weave not available. Install with: uv add 'wandb[weave]'")
                return False

    except Exception as e:
        logger.error(f"Failed to initialize Weave: {e}")
        return False


def is_weave_initialized() -> bool:
    """Check if Weave has been initialized."""
    return _weave_initialized


def get_weave_op():
    """
    Get the weave.op decorator.

    Returns:
        The weave.op decorator or None if Weave is not available

    Example:
        >>> weave_op = get_weave_op()
        >>> if weave_op:
        ...     @weave_op
        ...     def my_function(x):
        ...         return x * 2
    """
    if not _weave_initialized:
        return None

    try:
        import wandb.weave as wandb_weave
        return wandb_weave.op
    except ImportError:
        try:
            import weave
            return weave.op
        except ImportError:
            logger.warning("Weave.op decorator not available")
            return None


def weave_op_decorator(func=None, **kwargs):
    """
    Decorator factory that applies @weave.op() if Weave is available.

    This decorator is safe to use even when Weave is not initialized or available.
    The function will work normally, just without tracing.

    Args:
        func: Function to decorate
        **kwargs: Additional arguments passed to weave.op()

    Example:
        >>> @weave_op_decorator
        ... def my_function(x):
        ...     return x * 2
    """
    def decorator(f):
        # Try to get weave.op decorator
        try:
            import wandb.weave as wandb_weave
            op_decorator = wandb_weave.op
            return op_decorator(**kwargs)(f)
        except ImportError:
            try:
                import weave
                op_decorator = weave.op
                return op_decorator(**kwargs)(f)
            except ImportError:
                # Weave not available, return function as-is
                return f

    if func is not None:
        # Called as @weave_op_decorator without arguments
        return decorator(func)
    else:
        # Called as @weave_op_decorator(**kwargs)
        return decorator


def weave_op_decorator_configured(config_key: str = None):
    """
    WeaveConfigに基づいてトレースするデコレータ

    Args:
        config_key: WeaveConfigのキー（例: "TRACE_EMBEDDINGS"）

    Returns:
        デコレータ関数

    Example:
        >>> @weave_op_decorator_configured("TRACE_EMBEDDINGS")
        ... def encode(texts):
        ...     return embeddings
    """
    def decorator(func):
        # WeaveConfigから設定を取得
        from signate_studentcup_2025.config import WeaveConfig

        # Weaveが有効かつ、特定の設定がTrueの場合のみトレース
        if config_key and getattr(WeaveConfig, config_key, False):
            return weave_op_decorator(func)
        else:
            return func

    return decorator
