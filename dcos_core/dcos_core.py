import math
import numpy as np


class Sample:
    __slots__ = ["level", "time"]
    def __init__(self, level, time):
        self.level = level
        self.time = time


class Price:
    __slots__ = ["bid", "ask", "time"]
    def __init__(self, bid, ask, time):
        self.bid = bid
        self.ask = ask
        self.time = time
    def getMid(self):
        return (self.bid + self.ask) / 2


class DcOS:
    __slots__ = [
        "threshold",
        "mode",
        "extreme",
        "prevExtreme",
        "reference",
        "prevDC",
        "DC",
        "osL",
        "totalMove",
        "nOS",
        "dcL",
        "nDC",
        "osSegment",
    ]

    def __init__(self, threshold, initialMode=0, midpriceMode=False):
        self.midpriceMode = midpriceMode
        self.threshold = threshold
        self.mode = initialMode
        self.extreme = self.prevExtreme = self.reference = self.prevDC = self.DC = Sample(0, 0)
        self.osL = 0
        self.totalMove = 0
        self.nOS = 0
        self.nDC = 0
        self.dcL = 0
        self.osSegment = []

    def run(self, sample):
        if self.midpriceMode:
            current_price = Sample(sample.getMid(), sample.time)
        else:
            current_price = Sample(sample.level, sample.time)

        # initialize state on first tick
        if self.extreme.level == 0:
            self.extreme = self.prevExtreme = self.reference = self.prevDC = self.DC = current_price
            return 0

        # unbiased symmetric threshold in log space
        eta = math.log1p(self.threshold)  # same positive value for up and down

        # neutral start: infer mode after first threshold-sized move, emit no event
        if self.mode == 0:
            d = math.log(current_price.level / self.reference.level)
            if abs(d) >= eta:
                self.mode = -1 if d > 0 else +1  # -1 means "up mode", +1 "down mode"
                self.extreme = self.reference = self.DC = self.prevDC = current_price
                self.nOS = 0
            return 0

        side = -self.mode  # +1 when in up mode, -1 when in down mode

        # price continues with current mode -> possible overshoot
        if current_price.level * side > side * self.extreme.level:
            self.extreme = current_price
            if side * math.log(self.extreme.level / self.reference.level) >= eta:
                self.reference = self.extreme
                self.nOS += 1
                return 2 * side
            return 0

        # price moves against current mode -> possible directional change
        self.dcL = -side * math.log(current_price.level / self.extreme.level)
        if self.dcL >= eta:
            self.osSegment.append(abs(math.log(self.extreme.level / self.DC.level)))
            self.osL = side * math.log(self.extreme.level / self.DC.level)
            self.totalMove = side * math.log(self.extreme.level / self.prevExtreme.level)
            self.prevDC = self.DC
            self.DC = current_price
            self.prevExtreme = self.extreme
            self.extreme = self.reference = current_price
            self.mode *= -1
            self.nOS = 0
            self.nDC += 1
            return -side
        return 0
