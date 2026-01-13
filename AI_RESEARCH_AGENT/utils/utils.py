def clean_text(text: str) -> str:
    """Basic text preprocessing: lowercase, strip spaces, remove extra whitespace"""
    text = text.lower().strip()
    text = " ".join(text.split())
    return text

def split_data(data: list, train_ratio=0.8, val_ratio=0.1):
    """Split a list into train, validation, and test sets"""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, val, test