import datetime
import typing


class NoisyNumber:
    latest: float
    filtered: float
    updated: datetime

    # weight of old value
    weight: float

    # a jump of more than this value will instantly overwrite filtered value
    jump: float

    # the amount of change to trigger an event
    last_event_value: float
    trigger_threshold: float
    trigger_callback: typing.Callable

    def __init__(self, value: float, weight=0.75, jump=None):
        self.latest = self.filtered = value
        self.weight = min(1.0, max(0.0, weight))
        self.jump = jump
        self.updated = datetime.datetime.now()
        self.trigger_threshold = 0.0
        self.trigger_callback = None
        self.last_event_value = value

    def set(self, v: float, trigger: bool = False):
        self.latest = self.filtered = v
        if trigger and self.trigger_callback:
            self.trigger_callback(self.filtered)
            self.last_event_value = self.filtered

    def filter(self, v: float):
        now = datetime.datetime.now()
        if (now - self.updated).seconds > 5.0 or (self.jump and abs(v - self.filtered) > self.jump):
            # no filtering, take the new value as-is
            self.latest = self.filtered = v
        else:
            # filter through low pass
            self.latest = v
            self.filtered = self.filtered * self.weight + v * (1.0 - self.weight)
        # check trigger
        if self.trigger_callback and abs(self.last_event_value - self.filtered) > self.trigger_threshold:
            self.trigger_callback(self.filtered)
            self.last_event_value = self.filtered
        self.updated = datetime.datetime.now()

    def on_trigger(self, threshold: float, callback: typing.Callable):
        self.trigger_threshold = threshold
        self.trigger_callback = callback
