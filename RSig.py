import iisignature


class RSig:
    """
    Class handling computing (possibly randomised) signatures
    """

    def __init__(self, streams=None):
        self.streams = streams # hold the original streams as a list of numpy arrays
        self.sigs = [] # hold the list of signatures computed

    def sig_truncated(self, level: int):
        """
        Compute truncated signature to level
        :param level: level of truncation of the signature
        :return: The signature of the stream held
        """
        for stream in self.streams:
            self.sigs.append(iisignature.sig(stream, level))

    def load_stream(self, streams):
        self.streams = streams
