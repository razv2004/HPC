class User:
    def __init__(self, name, share):
        self.name = name
        self.share = share

    def __str__(self):
        return f"U({self.name},{self.share})"
