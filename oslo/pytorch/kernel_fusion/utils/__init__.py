def is_iterable(elem):
    try:
        iter(elem)
        return True
    except:
        return False
