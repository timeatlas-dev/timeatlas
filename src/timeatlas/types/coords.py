
class Coords:
    """ Defines geographic coordinates

    The format to use is decimal degrees (DD). For instance:

        46.9491463,7.4388499

    For the train station of Bern, the capital of Switzerland.
    """

    def __init__(self, lat: float, long: float):
        self.lat = lat
        self.long = long

    def __repr__(self):
        return "{}, {}".format(self.lat, self.long)
