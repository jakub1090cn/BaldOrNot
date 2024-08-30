class BaldOrNotDataError(Exception):
    """Custom exception raised when an image fails to load."""

    def __init__(self, message="Failed to load image. Image is None."):
        self.message = message
        super().__init__(self.message)
