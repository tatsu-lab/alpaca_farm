# maps to common.py
def apex_is_installed():
    try:
        import apex

        return True
    except ImportError as _:
        return False
