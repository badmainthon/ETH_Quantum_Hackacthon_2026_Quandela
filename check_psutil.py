try:
    import psutil
    print('psutil available')
except ImportError:
    print('psutil not available')
