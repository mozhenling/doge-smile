import sys

class Tee:
    """
    It mimics the behavior of the Unix `tee` command.
    It sends output to both terminal and a file.
    Good for debugging or keeping output logs.
    """
    def __init__(self, fname, mode="a"):
        # Save the current standard output (usually the terminal)
        self.stdout = sys.stdout
        # Open a file to write to, defaulting to append mode
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)  # Write to terminal
        self.file.write(message)  # Also write to file
        self.flush()  # Flush both outputs

    def flush(self):
        """Flushing ensures that the output is immediately written, not buffered"""
        self.stdout.flush()  # Flush the terminal output
        self.file.flush()  # Flush the file buffer